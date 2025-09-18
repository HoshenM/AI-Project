import platform
import psutil
import subprocess
import sys
from datetime import datetime

def get_hardware_info():
    """
    Comprehensive hardware information detection
    """
    hardware_info = {}

    # System Information
    hardware_info['System'] = {
        'OS': platform.system(),
        'OS Version': platform.version(),
        'Architecture': platform.architecture()[0],
        'Machine': platform.machine(),
        'Processor': platform.processor(),
        'Python Version': sys.version
    }

    # CPU Information
    hardware_info['CPU'] = {
        'Physical Cores': psutil.cpu_count(logical=False),
        'Total Cores': psutil.cpu_count(logical=True),
        'Max Frequency': f"{psutil.cpu_freq().max:.2f} MHz" if psutil.cpu_freq() else "N/A",
        'Current Frequency': f"{psutil.cpu_freq().current:.2f} MHz" if psutil.cpu_freq() else "N/A"
    }

    # Memory Information
    svmem = psutil.virtual_memory()
    hardware_info['Memory'] = {
        'Total RAM': f"{svmem.total / (1024**3):.2f} GB",
        'Available RAM': f"{svmem.available / (1024**3):.2f} GB",
        'Used RAM': f"{svmem.used / (1024**3):.2f} GB",
        'RAM Usage %': f"{svmem.percent}%"
    }

    # Disk Information
    disk_info = []
    partitions = psutil.disk_partitions()
    for partition in partitions:
        try:
            partition_usage = psutil.disk_usage(partition.mountpoint)
            disk_info.append({
                'Device': partition.device,
                'Mountpoint': partition.mountpoint,
                'File System': partition.fstype,
                'Total Size': f"{partition_usage.total / (1024**3):.2f} GB",
                'Used': f"{partition_usage.used / (1024**3):.2f} GB",
                'Free': f"{partition_usage.free / (1024**3):.2f} GB"
            })
        except PermissionError:
            continue
    hardware_info['Disk'] = disk_info

    # GPU Information (if available)
    gpu_info = get_gpu_info()
    if gpu_info:
        hardware_info['GPU'] = gpu_info

    return hardware_info

def get_gpu_info():
    """
    Get GPU information using nvidia-smi or other methods
    """
    gpu_info = {}

    try:
        # Try to get NVIDIA GPU info
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)

        gpu_lines = result.stdout.strip().split('\n')
        gpu_info['NVIDIA'] = []

        for i, line in enumerate(gpu_lines):
            name, memory, driver = line.split(', ')
            gpu_info['NVIDIA'].append({
                'GPU': i,
                'Name': name,
                'Memory': f"{memory} MB",
                'Driver Version': driver
            })

    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not available, try other methods
        pass

    # Try to detect other GPUs or integrated graphics
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'],
                                   capture_output=True, text=True)
            # Parse macOS GPU info (simplified)
            gpu_info['System'] = "macOS GPU detected (use Activity Monitor for details)"

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return gpu_info

def log_hardware_info():
    """
    Log hardware information to console and file
    """
    print("\n" + "="*60)
    print("  Hardware Inforation")
    print("="*60)

    hardware_info = get_hardware_info()

    # Print to console
    for category, info in hardware_info.items():
        print(f"\n {category.upper()}:")
        if isinstance(info, dict):
            for key, value in info.items():
                print(f"   {key}: {value}")
        elif isinstance(info, list):
            for i, item in enumerate(info):
                print(f"   {category} {i+1}:")
                for key, value in item.items():
                    print(f"      {key}: {value}")

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"hardware_info_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("Hardware Information Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for category, info in hardware_info.items():
            f.write(f"\n{category.upper()}:\n")
            f.write("-" * 20 + "\n")
            if isinstance(info, dict):
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")
            elif isinstance(info, list):
                for i, item in enumerate(info):
                    f.write(f"{category} {i+1}:\n")
                    for key, value in item.items():
                        f.write(f"  {key}: {value}\n")

    print(f"\n Hardware information saved to: {filename}")
    return hardware_info, filename

def monitor_resource_usage():
    """
    Get current resource usage snapshot
    """
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent

# Detect Hardware Configuration
print("\n Detecting Hardware Configuration...")
hardware_info, hardware_file = log_hardware_info()
project_start_time = time.time()

