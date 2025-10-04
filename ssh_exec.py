import argparse
import paramiko
import sys

HOST = "94.103.12.81"
PASSWORD = "Y?XgmvQzV&Mf"
USER = "root"


def safe_print(text: str) -> None:
    # Ensure we can print even if console encoding lacks certain characters
    print(text.encode("ascii", errors="replace").decode("ascii"))


def run_commands(commands):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=20)

    try:
        for cmd in commands:
            safe_print(f"\n=== Running: {cmd} ===")
            stdin, stdout, stderr = client.exec_command(cmd, get_pty=True)
            stdout_data = stdout.read().decode(errors="replace")
            stderr_data = stderr.read().decode(errors="replace")
            if stdout_data:
                safe_print(stdout_data)
            if stderr_data:
                safe_print("[stderr]")
                safe_print(stderr_data)
    finally:
        client.close()


def main(argv):
    parser = argparse.ArgumentParser(description="Execute commands over SSH")
    parser.add_argument("commands", nargs="+", help="Commands to run on remote host")
    args = parser.parse_args(argv)

    run_commands(args.commands)


if __name__ == "__main__":
    main(sys.argv[1:])
