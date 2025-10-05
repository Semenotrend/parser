import paramiko
from pathlib import Path

HOST = "94.103.12.81"
USER = "root"
PASSWORD = "Y?XgmvQzV&Mf"


def upload(local_path: Path, remote_path: str) -> None:
    client = paramiko.Transport((HOST, 22))
    client.connect(username=USER, password=PASSWORD)
    try:
        sftp = paramiko.SFTPClient.from_transport(client)
        sftp.put(str(local_path), remote_path)
    finally:
        client.close()


def main():
    root = Path(__file__).resolve().parent
    mapping = {
        root / ".env": "/home/Jeka/projects/parser/.env",
        root / "channels.txt": "/home/Jeka/projects/parser/channels.txt",
        root / "79380845060.session": "/home/Jeka/projects/parser/79380845060.session",
    }
    for local, remote in mapping.items():
        if not local.exists():
            print(f"Skip {local}: not found")
            continue
        print(f"Uploading {local} -> {remote}")
        upload(local, remote)
    print("Done")

if __name__ == "__main__":
    main()
