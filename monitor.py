import threading, subprocess, time, psutil

def monitor_idle_gpu_cpu(duration=10, interval=1):
    
    power_values = []
    gpu_util_values = []
    cpu_util_values = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        power, gpu_util = map(float, result.stdout.strip().split(", "))
        cpu_util = psutil.cpu_percent(interval=0.1)
        
        power_values.append(power)
        gpu_util_values.append(gpu_util)
        cpu_util_values.append(cpu_util)
        
        time.sleep(interval)
    
    avg_power = sum(power_values) / len(power_values)
    avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values)
    avg_cpu_util = sum(cpu_util_values) / len(cpu_util_values)
    
    return avg_power, avg_gpu_util, avg_cpu_util

avg_time = 10
avg_power, avg_gpu_util, avg_cpu_util = monitor_idle_gpu_cpu(duration=avg_time)

print(f'\nAverage values over {avg_time} seconds: \nAVG_GPU_POWER = {avg_power}, AVG_GPU_UTIL = {avg_gpu_util}, AVG_CPU_UTIL = {avg_cpu_util}\n')

monitoring = True

def monitor_gpu(log_file = 'gpu_usage_log.csv', interval = 1):

    query_params = [
        "timestamp", "power.draw", "memory.used", "memory.total",
        "utilization.gpu", "utilization.memory", "temperature.gpu",
        "fan.speed", "clocks.sm", "clocks.gr"
    ]
    
    query_str = ",".join(query_params)
    
    with open(log_file, "w") as f:
        f.write("Timestamp,Power (W),Memory Used (MB),Memory Total (MB),GPU Util (%),"
                "Memory Util (%),Temp (C),Fan Speed (%),Clock SM (MHz),Clock Mem (MHz),"
                "CPU Usage (%)\n")
    
    while monitoring:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query_str, "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )

        gpu_data = list(map(float, result.stdout.strip().split(", ")[1:]))
        gpu_data[0] = gpu_data[0] - avg_power
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent() - avg_cpu_util
        
        log_entry = f"{timestamp}," + ",".join(list(map(str, gpu_data))) + f",{cpu_usage}\n"
        
        with open(log_file, "a") as f:
            f.write(log_entry)

        time.sleep(interval)

monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_TRANSFORMER.csv', 0.01), daemon=True)
monitor_thread.start()

#
# BURAYA MONITOR ETMEK ISTEDIGIMIZ KOD
#

monitoring = False
