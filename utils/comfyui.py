import json
import urllib.request
import urllib.parse
import os
import sys
import io
import mimetypes
import time
from PIL import Image
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from .core import is_image_file

class ComfyUIClient:
    def __init__(self, server_address="http://127.0.0.1:8188", poll_timeout=300):
        self.server_address = server_address
        self.client_id = "python_comfyui_client"
        self.poll_timeout = poll_timeout
    
    def _print(self, msg, progress_bar=None):
        if progress_bar:
            progress_bar.write(msg)
        else:
            print(msg)
    
    def upload_image(self, image_path, progress_bar=None):
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        image_name = os.path.basename(image_path)
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "image/jpeg"
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{image_name}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode("utf-8") + image_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="subfolder"\r\n\r\n\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
            f"--{boundary}--\r\n"
        ).encode("utf-8")
        
        url = f"{self.server_address}/upload/image"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Content-Length", len(body))
        
        try:
            response = urllib.request.urlopen(req, body)
            result = json.loads(response.read().decode("utf-8"))
            return result.get("name", image_name)
        except Exception as e:
            self._print(f"上传图片失败: {e}", progress_bar)
            return None
    
    def load_workflow(self, workflow_path):
        with open(workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def update_workflow_with_image(self, workflow, image_name):
        updated_workflow = json.loads(json.dumps(workflow))
        for node_id, node_data in updated_workflow.items():
            if node_data.get("class_type") == "LoadImage":
                if "inputs" in node_data:
                    node_data["inputs"]["image"] = image_name
        return updated_workflow
    
    def execute_workflow(self, workflow, progress_bar=None):
        data = {
            "prompt": workflow,
            "client_id": self.client_id
        }
        
        url = f"{self.server_address}/prompt"
        req = urllib.request.Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        
        try:
            response = urllib.request.urlopen(req, json.dumps(data).encode("utf-8"))
            result = json.loads(response.read().decode("utf-8"))
            prompt_id = result.get("prompt_id")
            
            if not prompt_id:
                self._print("错误：未返回prompt ID", progress_bar)
                return []
            
            return self.get_output_images(prompt_id, progress_bar)
        except Exception as e:
            self._print(f"执行工作流失败: {e}", progress_bar)
            return []
    
    def get_output_images(self, prompt_id, progress_bar=None, poll_timeout=None):
        poll_timeout = self.poll_timeout if poll_timeout is None else poll_timeout
        url = f"{self.server_address}/history/{prompt_id}"
        start_time = time.time()
        await_str = f"等待 ComfyUI 输出 (prompt: {prompt_id}) ..."

        # poll_timeout<=0 表示不限制，等价于无限等待（仍保留异常重试）
        while True:
            if poll_timeout > 0 and time.time() - start_time > poll_timeout:
                self._print(f"等待 ComfyUI 结果超时（{int(poll_timeout)}s）：{prompt_id} | {await_str}", progress_bar)
                return []
            try:
                response = urllib.request.urlopen(url)
                history = json.loads(response.read().decode("utf-8"))
                
                if prompt_id in history:
                    outputs = history[prompt_id].get("outputs", {})
                    images = []
                    
                    for node_id, node_output in outputs.items():
                        if "images" in node_output:
                            for img_data in node_output["images"]:
                                img_url = f"{self.server_address}/view?filename={urllib.parse.quote(img_data['filename'])}&subfolder={urllib.parse.quote(img_data.get('subfolder', ''))}&type={urllib.parse.quote(img_data.get('type', 'output'))}"
                                img_response = urllib.request.urlopen(img_url)
                                img = Image.open(io.BytesIO(img_response.read()))
                                images.append((img, img_data))
                    
                    return images
            except Exception:
                pass
            
            time.sleep(1)
    
    def save_images(self, images, save_dir="comfyui_output", original_filename=None, progress_bar=None):
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []
        
        for i, (image, metadata) in enumerate(images):
            if original_filename and i == 0:
                filename = original_filename
            else:
                filename = metadata.get("filename", f"output_{i}.png")
            
            save_path = os.path.join(save_dir, filename)
            image.save(save_path)
            saved_paths.append(save_path)
            self._print(f"  已保存图片: {save_path}", progress_bar)
        
        return saved_paths
    
    def process_single_image(self, image_path, workflow_path, output_dir="comfyui_output", progress_bar=None):
        self._print(f"处理: {os.path.basename(image_path)}", progress_bar)
        
        if not os.path.exists(image_path):
            self._print(f"错误：图片不存在于 {image_path}", progress_bar)
            return []
        
        if not os.path.exists(workflow_path):
            self._print(f"错误：工作流JSON不存在于 {workflow_path}", progress_bar)
            return []
        
        self._print("  正在上传图片到ComfyUI...", progress_bar)
        uploaded_image_name = self.upload_image(image_path, progress_bar)
        if not uploaded_image_name:
            self._print("  上传图片失败", progress_bar)
            return []
        self._print(f"  已上传图片: {uploaded_image_name}", progress_bar)
        
        self._print("  正在加载工作流...", progress_bar)
        workflow = self.load_workflow(workflow_path)
        
        self._print("  正在更新工作流中的图片...", progress_bar)
        workflow = self.update_workflow_with_image(workflow, uploaded_image_name)
        
        self._print("  正在执行工作流...", progress_bar)
        output_images = self.execute_workflow(workflow, progress_bar)
        
        self._print(f"  已接收 {len(output_images)} 张输出图片", progress_bar)
        
        self._print("  正在保存图片...", progress_bar)
        original_filename = os.path.basename(image_path)
        saved_paths = self.save_images(output_images, output_dir, original_filename, progress_bar)
        
        return saved_paths
    
    def process_batch_images(self, image_paths, workflow_path, output_dir="comfyui_output"):
        all_saved_paths = []
        
        with tqdm(total=len(image_paths), desc="处理进度", unit="张", leave=True) as pbar:
            for image_path in image_paths:
                saved_paths = self.process_single_image(image_path, workflow_path, output_dir, pbar)
                all_saved_paths.extend(saved_paths)
                pbar.update(1)
        
        return all_saved_paths

def main():
    comfy_client = ComfyUIClient()
    
    input_image_path = r"imgs"
    workflow_json_path = "workflows/f2k-漫画去码-py.json"
    output_dir = r"runs\comfyui_output"
    
    saved_paths = []
    
    if os.path.isfile(input_image_path):
        if is_image_file(input_image_path):
            saved_paths = comfy_client.process_single_image(input_image_path, workflow_json_path, output_dir)
        else:
            print(f"错误：{input_image_path} 不是有效的图片文件")
    
    elif os.path.isdir(input_image_path):
        image_paths = [
            os.path.join(input_image_path, f)
            for f in os.listdir(input_image_path)
            if os.path.isfile(os.path.join(input_image_path, f)) and is_image_file(f)
        ]
        if image_paths:
            saved_paths = comfy_client.process_batch_images(image_paths, workflow_json_path, output_dir)
        else:
            print(f"错误：目录 {input_image_path} 中没有有效的图片文件")
    
    else:
        print(f"错误：源文件/目录 {input_image_path} 不存在")
        return
    
    print("\n=== 完成！ ===")
    print(f"共 {len(saved_paths)} 张图片已保存到: {output_dir}")

if __name__ == "__main__":
    main()