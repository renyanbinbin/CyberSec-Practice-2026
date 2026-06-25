import socket
import sys
import os
import time

# 常量定义
SERV_PORT = 34000
SERV_ADDR = "127.0.0.1"
BUFFER_SIZE = 1048576  # 1MB缓冲区
EXIT_FLAG = "+++"
OUTPUT_DIR = os.path.realpath("./received_files")  # 限制文件写入目录，防路径穿越
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB 文件大小上限，防磁盘耗尽
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".csv", ".json", ".xml", ".md", ".py", ".zip", ".log"}
RECV_TIMEOUT = 30  # 30秒接收超时

# 工具函数
def print_welcome():
    """打印欢迎信息"""
    print("=" * 50)
    print(" Simple TCP File Client ")
    print("=" * 50)
    print(f"Server Address: {SERV_ADDR}:{SERV_PORT}")
    print(f"Enter '{EXIT_FLAG}' to exit")
    print("=" * 50)


def validate_filename(filename):
    """
    检查文件名是否合法，使用 realpath 防止路径穿越攻击。
    修复：原代码仅检查 ".." 字符串，可用 URL 编码等方式绕过。
          现改用 os.path.realpath() 确保文件始终落在 OUTPUT_DIR 内。
    """
    if not filename:
        print("Error: filename is empty")
        return False

    # 禁止空字符和绝对路径
    if "\x00" in filename or filename.startswith("/") or filename.startswith("\\"):
        print("Error: invalid filename")
        return False

    # 使用 realpath 防路径穿越
    full_path = os.path.realpath(os.path.join(OUTPUT_DIR, os.path.basename(filename)))
    if not full_path.startswith(OUTPUT_DIR + os.sep):
        print("Error: path traversal detected")
        return False

    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        print(f"Error: file extension '{ext}' not allowed")
        return False

    return True


def create_file(filename):
    """
    创建文件用于写入，写入前确保目录存在。
    修复：原代码使用裸 except Exception，现替换为具体异常类型。
    """
    try:
        full_path = os.path.realpath(os.path.join(OUTPUT_DIR, os.path.basename(filename)))
        os.makedirs(OUTPUT_DIR, exist_ok=True)  # 确保输出目录存在
        fp = open(full_path, "wb")
        print(f"[INFO] File '{filename}' opened successfully")
        return fp
    except FileNotFoundError:
        print(f"[ERROR] Directory not found for file '{filename}'")
        return None
    except PermissionError:
        print(f"[ERROR] Permission denied for file '{filename}'")
        return None
    except OSError as e:
        print(f"[ERROR] Cannot open file '{filename}': {str(e)}")
        return None


def receive_file(sockfd, fp):
    """
    接收文件内容，加入大小限制防止磁盘耗尽（拒绝服务）。
    修复：原代码无文件大小上限，现限制 MAX_FILE_SIZE（100MB）。
    """
    total_bytes = 0
    start_time = time.time()

    while True:
        try:
            data = sockfd.recv(BUFFER_SIZE)
        except socket.timeout:
            print("[WARNING] Receive timeout")
            break
        except OSError as e:
            print(f"[ERROR] Socket error during receive: {str(e)}")
            break

        if not data:
            break

        # 文件大小检查，防止恶意服务端发送无限数据
        if total_bytes + len(data) > MAX_FILE_SIZE:
            print(f"[WARNING] File exceeds maximum size ({MAX_FILE_SIZE // 1024 // 1024}MB), truncating")
            fp.write(data[:MAX_FILE_SIZE - total_bytes])
            total_bytes = MAX_FILE_SIZE
            break

        fp.write(data)
        total_bytes += len(data)

    end_time = time.time()
    duration = end_time - start_time

    return total_bytes, duration


def print_stats(filename, total_bytes, duration):
    """打印接收统计信息"""
    print("\n[RESULT]")
    print(f"File Name     : {filename}")
    print(f"Bytes Received: {total_bytes}")

    if duration > 0:
        speed = total_bytes / duration / 1024  # KB/s
        print(f"Time Used     : {duration:.2f} s")
        print(f"Speed         : {speed:.2f} KB/s")

    print("=" * 50)


def connect_server():
    """
    创建并连接socket。
    修复：原代码使用裸 except Exception，现替换为具体异常类型。
    """
    try:
        sockfd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sockfd.settimeout(RECV_TIMEOUT)  # 设置接收超时
        print(f"[DEBUG] Socket FD: {sockfd.fileno()}")

        sockfd.connect((SERV_ADDR, SERV_PORT))
        print(f"[INFO] Connected to server {SERV_ADDR}:{SERV_PORT}")

        return sockfd

    except ConnectionRefusedError:
        print("[ERROR] Connection refused. Server may be down.")
    except socket.gaierror:
        print("[ERROR] Invalid IP address")
    except socket.timeout:
        print("[ERROR] Connection attempt timed out")
    except OSError as e:
        print(f"[ERROR] Socket error: {str(e)}")

    return None


# 主逻辑
def main():
    print_welcome()

    sockfd = connect_server()
    if not sockfd:
        return

    try:
        while True:
            filename = input("\nPlease enter the required document: ").strip()

            # 退出条件
            if filename == EXIT_FLAG:
                print("[INFO] Exit command received")
                break

            # 文件名校验（含路径穿越防护）
            if not validate_filename(filename):
                continue

            # 发送文件名
            try:
                sockfd.sendall(filename.encode())
                print(f"[INFO] Sent filename: {filename}")
            except (OSError, ConnectionError) as e:
                print(f"[ERROR] Failed to send filename: {str(e)}")
                break

            # 创建文件
            fp = create_file(filename)
            if fp is None:
                continue

            # 接收文件（含大小限制）
            total_bytes, duration = receive_file(sockfd, fp)

            # 关闭文件
            try:
                fp.close()
            except OSError:
                pass

            # 判断服务器状态
            if total_bytes == 0:
                print("[WARNING] Server may have disconnected")
                break

            # 输出统计信息
            print_stats(filename, total_bytes, duration)

    finally:
        print("[INFO] Closing connection...")
        try:
            sockfd.close()
        except OSError:
            pass
        print("[INFO] Client exited successfully")


# 程序入口
if __name__ == "__main__":
    main()
