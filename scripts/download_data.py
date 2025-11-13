import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

# ------------------- 全局优化配置（可根据网络调整） -------------------
CHUNK_SIZE = 64 * 1024 * 1024  # 64MB分块
MAX_WORKERS = 12  # 12并发
TIMEOUT = 120  # 超时时间2分钟
RETRY_TIMES = 3  # 失败自动重试3次

# 抑制SSL警告（避免日志刷屏）
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# 创建requests会话（复用连接，减少TCP握手开销）
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    max_retries=RETRY_TIMES,
    pool_connections=MAX_WORKERS * 2,  # 连接池大小=并发数×2，确保并行生效
    pool_maxsize=MAX_WORKERS * 2
)
session.mount('https://', adapter)
session.mount('http://', adapter)

# ------------------- 优化后的核心下载函数 -------------------
def download_single_file(url, save_path):
    """单文件下载：支持断点续传、进度条、连接复用、自动重试"""
    # 创建保存目录（如果不存在）
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"📂 创建保存目录：{save_dir}")
    
    # 断点续传逻辑
    resume_size = 0
    if os.path.exists(save_path):
        resume_size = os.path.getsize(save_path)
        
        # 先获取远程文件大小以检查是否已完整下载
        try:
            response = session.head(url, allow_redirects=True, timeout=TIMEOUT)
            total_size = int(response.headers.get('content-length', 0))
            
            # 如果本地文件大小等于远程文件大小，表示已完整下载
            if total_size > 0 and resume_size == total_size:
                print(f"✅ {os.path.basename(save_path)} 已完整下载，跳过")
                return True  # 跳过下载
        except Exception as e:
            print(f"⚠️ 获取文件大小失败: {e}，继续断点续传")
        
        print(f"🔄 检测到{os.path.basename(save_path)}已下载 {resume_size/1024/1024:.1f}MB，继续下载...")
    
    headers = {"Range": f"bytes={resume_size}-"} if resume_size > 0 else {}
            
    file_name = os.path.basename(save_path)
    print(f"📥 开始下载 {file_name}")
        
    try:
        # 用复用会话发送请求（替代原requests.get，速度更快）
        response = session.get(
            url, 
            stream=True, 
            timeout=TIMEOUT, 
            headers=headers,
            verify=False  # 保留跳过SSL验证（适配部分网络）
        )
        response.raise_for_status()  # 链接无效时抛出错误
        
        # 修复：正确获取总大小（处理content-length为空的情况）
        content_length = response.headers.get('content-length')
        if content_length is None or not content_length.strip():
            total_size = None  # 未知大小
        else:
            remaining_size = int(content_length)
            total_size = resume_size + remaining_size  # 总字节数

        # 修复进度条：用字节数作为基准，避免单位转换混乱
        with open(save_path, 'ab') as file, tqdm(
            desc=file_name,
            total=total_size,  # 字节数（tqdm自动处理单位）
            initial=resume_size,  # 已下载字节数
            unit='B',  # 基准单位为字节
            unit_scale=True,  # 自动缩放单位（B→KB→MB→GB）
            unit_divisor=1024,
            ncols=80,
            leave=True,
            dynamic_ncols=True  # 自适应终端宽度
        ) as bar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    file.write(chunk)
                    bar.update(len(chunk))
        
        print(f"\n✅ {file_name} 下载完成！")
        return True
    
    except Exception as e:
        print(f"\n❌ {file_name} 下载失败：{str(e)}")
        return False

# ------------------- 批量多线程下载（保留你的原有逻辑） -------------------
def batch_download(files_list, max_workers=MAX_WORKERS):
    """批量多线程下载：max_workers=同时下载的文件数"""
    print(f"📥 开始批量下载（并发数：{max_workers}，分块大小：{CHUNK_SIZE/1024/1024:.0f}MB）")
    print(f"📋 共需下载 {len(files_list)} 个文件\n")
    
    # 多线程执行（依赖会话连接池，真正并行）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda x: download_single_file(x[0], x[1]),
            files_list
        ))
    
    # 统计结果
    success_count = results.count(True)
    fail_count = results.count(False)
    print(f"\n📊 下载统计：成功 {success_count} 个 | 失败 {fail_count} 个")
    if fail_count > 0:
        print("🔄 失败文件可直接重新运行脚本，支持断点续传")

if __name__ == "__main__":
    # ------------------- 保留你的文件路径和链接配置 -------------------
    files_to_save_folder = "../data/raw/"
    files_to_download = [
        # ("https://figshare.com/ndownloader/files/43293162?download=1", f"{files_to_save_folder}RGD1_RPs.h5"),
        # ("https://figshare.com/ndownloader/files/43291989?download=1", f"{files_to_save_folder}RandP_smiles.txt"),
        ("https://figshare.com/ndownloader/files/38170323?download=1", f"{files_to_save_folder}RGD1_CHNO.h5"),
        ("https://figshare.com/ndownloader/files/36035789?download=1", f"{files_to_save_folder}Transition1x.h5"),
        ("https://zenodo.org/records/11493786/files/Reactions.tar.bz2?download=1", f"{files_to_save_folder}RDB19-Rad_Reactions.tar.bz2")
        # 可继续添加更多文件...
    ]
    
    # 开始下载（使用你设置的MAX_WORKERS）
    batch_download(files_to_download, max_workers=MAX_WORKERS)