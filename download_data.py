import os
import subprocess
import sys

def download_afhq():
    """
    Download AFHQ dataset thru StarGAN v2 official repository.
    """
    print("🚀 Start downloading StarGAN v2 AFHQ dataset...")

    if not os.path.exists('stargan-v2'):
        print("1. Cloning StarGAN v2 repository...")
        subprocess.run(['git', 'clone', 'https://github.com/clovaai/stargan-v2.git'], check=True)
    else:
        print("1. Repository already exists.")

    print("2. Compiling AFHQ dataset download script... (it might takes time)")
    
    os.chdir('stargan-v2')
    try:
        subprocess.run(['bash', 'download.sh', 'afhq-dataset'], check=True)
        print("\n✅ Download Completed! Check ./stargan-v2/data/afhq directory.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error occurred while downloading: {e}")
        print("Warning: Windows users should use Git Bash etc.")
    except FileNotFoundError:
        print("\n❌ Can't find 'bash' command. Use Linux/Mac env. or WSL.")
    finally:
        os.chdir('..')

if __name__ == "__main__":
    download_afhq()
