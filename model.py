import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
from scipy.interpolate import interp1d
import time

# ===== 0 中文字体 & Matplotlib 全局设置 =====
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 
                                   'PingFang SC', 'Noto Sans CJK SC',
                                   'Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ===== 1 读取数据 =====
CSV_FILE = 'temperature.csv'
if not Path(CSV_FILE).exists():
    sys.exit(f'❌ 未找到 {CSV_FILE} ，请与脚本放在同一目录！')

df = pd.read_csv(CSV_FILE, names=['time','temperature','volte'], header=0)

t = df['time'].astype(float).to_numpy()
y = df['temperature'].astype(float).to_numpy()
u = df['volte'].astype(float).to_numpy()

# 采样周期
dt = np.round(np.mean(np.diff(t)), 6)
R = 3.5  # 阶跃电压变化量

# ===== 2 两点法辨识 =====
Y0, Yss = y[0], y[-1]
K = (Yss - Y0) / R

# 28.3% 与 63.2% 点
Y28, Y63 = Y0 + 0.283*(Yss-Y0), Y0 + 0.632*(Yss-Y0)

# 使用插值精确找到对应时间点
f = interp1d(t, y, kind='linear')
t1 = np.linspace(t.min(), t.max(), 10000)
y1 = f(t1)

idx28 = np.argmax(y1 >= Y28)
idx63 = np.argmax(y1 >= Y63)
T1, Y1 = t1[idx28], y1[idx28]
T2, Y2 = t1[idx63], y1[idx63]

# 计算参数
M1 = np.log(1 - (Y1 - Y0)/(K * R))
M2 = np.log(1 - (Y2 - Y0)/(K * R))
Tau = (T2 - T1)/(M1 - M2)
Lag = max((T2*M1 - T1*M2)/(M1 - M2), 0.01*Tau)  # 防止负值

# ===== 3 实际曲线与拟合曲线对比图 =====
def fopdt_resp(k, T, L, r, tvec, y0):
    """一阶加纯滞后模型响应"""
    yout = np.zeros_like(tvec)
    for i, tt in enumerate(tvec):
        if tt <= L:
            yout[i] = y0
        else:
            yout[i] = y0 + k*r*(1 - np.exp(-(tt-L)/T))
    return yout

# 生成模型响应
y_model = fopdt_resp(K, Tau, Lag, R, t, Y0)

plt.figure(figsize=(8, 5))
plt.plot(t, y, 'r-', linewidth=2, label='实际温度曲线')
plt.plot(t, y_model, 'b--', linewidth=2, label='拟合模型曲线')
plt.axhline(Yss, ls='-.', c='g', alpha=0.7, label=f'稳态值 {Yss:.1f}℃')
plt.axhline(Y0, ls='-.', c='purple', alpha=0.7, label=f'初始值 {Y0:.1f}℃')
plt.scatter(T1, Y1, s=80, c='darkblue', zorder=5)
plt.scatter(T2, Y2, s=80, c='darkblue', zorder=5)
plt.annotate(f'28.3%点\n({T1:.1f}s, {Y1:.1f}℃)', (T1, Y1), xytext=(T1+5, Y1-0.5))
plt.annotate(f'63.2%点\n({T2:.1f}s, {Y2:.1f}℃)', (T2, Y2), xytext=(T2+5, Y2+0.5))

plt.xlabel('时间 (s)')
plt.ylabel('温度 (℃)')
plt.title('加热炉阶跃响应 - 模型拟合结果')
plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig('model_fit.png', dpi=200)
plt.show()

# ===== 4 PID 控制模块 =====
def simulate_pid(k, T, L, Kp, Ki, Kd, tvec, setpoint=35, u_min=0, u_max=10, y0=Y0):
    """带纯滞后的PID闭环仿真"""
    n = len(tvec)
    yk = y0
    ei = 0.0
    prev_error = setpoint - y0
    
    # 纯滞后队列
    delay_steps = max(1, int(round(L/dt)))
    dead = [0.0] * delay_steps
    
    yout = np.zeros(n)
    
    for i in range(n):
        # 计算误差
        err = setpoint - yk
        
        # PID计算
        P = Kp * err
        ei += err * dt
        I = Ki * ei
        de = (err - prev_error) / dt if i > 0 else 0
        D = Kd * de
        prev_error = err
        
        uk = P + I + D
        uk = np.clip(uk, u_min, u_max)
        
        # 更新纯滞后队列
        dead.append(uk)
        ueff = dead.pop(0)
        
        # 系统模型更新
        if i > 0:
            dy = (k * ueff - (yk - y0)) * dt / T
            yk += dy
        
        yout[i] = yk
    
    return yout

# ===== 5 PID 参数优化 =====
def pid_objective(params, k, T, L, tvec, setpoint=35, y0=Y0):
    """目标函数(最小化IAE和超调量)"""
    Kp, Ki, Kd = params
    y_sim = simulate_pid(k, T, L, Kp, Ki, Kd, tvec, setpoint, y0=y0)
    
    # 计算积分绝对误差(IAE)
    e = setpoint - y_sim
    iae = np.trapz(np.abs(e), tvec)
    
    # 计算超调量惩罚
    overshoot = max(np.max(y_sim) - setpoint, 0)
    overshoot_penalty = 100 * overshoot
    
    return iae + overshoot_penalty

def optimize_pid(k, T, L, tvec, setpoint=35, y0=Y0, 
                 n_particles=15, max_iter=20):
    """粒子群算法优化PID参数"""
    # 参数边界
    bounds = np.array([[0.1, 10.0], [0.001, 0.5], [0.1, 15.0]])
    
    # 初始化粒子群
    dim = 3
    particles = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], 
                                 size=(n_particles, dim))
    velocities = np.zeros((n_particles, dim))
    
    # 初始化最佳位置
    personal_best_pos = particles.copy()
    personal_best_fit = [pid_objective(p, k, T, L, tvec, setpoint, y0) 
                         for p in particles]
    
    global_best_idx = np.argmin(personal_best_fit)
    global_best_pos = personal_best_pos[global_best_idx]
    global_best_fit = personal_best_fit[global_best_idx]
    
    # PSO参数
    w = 0.7
    c1 = 1.5
    c2 = 1.5
    
    # 优化过程
    for iter in range(max_iter):
        for i in range(n_particles):
            # 更新速度和位置
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] + 
                            c1 * r1 * (personal_best_pos[i] - particles[i]) + 
                            c2 * r2 * (global_best_pos - particles[i]))
            
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
            
            # 计算新适应度
            current_fit = pid_objective(particles[i], k, T, L, tvec, setpoint, y0)
            
            # 更新最佳位置
            if current_fit < personal_best_fit[i]:
                personal_best_pos[i] = particles[i]
                personal_best_fit[i] = current_fit
                
                if current_fit < global_best_fit:
                    global_best_pos = particles[i]
                    global_best_fit = current_fit
    
    print(f"优化完成: Kp={global_best_pos[0]:.4f}, Ki={global_best_pos[1]:.6f}, Kd={global_best_pos[2]:.4f}")
    return global_best_pos

# ===== 6 PID 参数优化与对比 =====
# 初始PID参数
Kp_initial = 1.8
Ki_initial = 0.005
Kd_initial = 1.5

# 优化PID参数
print("\n正在优化PID参数...")
start_time = time.time()
Kp_opt, Ki_opt, Kd_opt = optimize_pid(K, Tau, Lag, t, setpoint=35, y0=Y0)
print(f"优化耗时: {time.time()-start_time:.2f}秒")

# ===== 7 闭环响应对比 =====
# 仿真闭环响应
y_initial = simulate_pid(K, Tau, Lag, Kp_initial, Ki_initial, Kd_initial, t)
y_opt = simulate_pid(K, Tau, Lag, Kp_opt, Ki_opt, Kd_opt, t)

plt.figure(figsize=(8, 5))
plt.plot(t, y_initial, 'r-', linewidth=1.5, label='初始PID')
plt.plot(t, y_opt, 'g-', linewidth=2.0, label='优化PID')
plt.axhline(35, color='k', linestyle='-.', alpha=0.7, label='设定值(35℃)')
plt.fill_between(t, 33.25, 36.75, color='gray', alpha=0.1)  # ±5%误差带

plt.xlabel('时间 (s)')
plt.ylabel('温度 (℃)')
plt.title('PID控制效果对比')
plt.grid(True, ls=':', alpha=0.7)
plt.legend()
plt.ylim(30, 40)
plt.tight_layout()
plt.savefig('pid_comparison.png', dpi=300)
plt.show()

# ===== 8 性能指标计算 =====
def calculate_metrics(y_sim, tvec, setpoint=35):
    """控制性能指标计算"""
    # 超调量
    overshoot = max(np.max(y_sim) - setpoint, 0)
    
    # 余差（稳态误差）
    steady_error = setpoint - np.mean(y_sim[-len(y_sim)//10:])
    
    # 5%回复时间
    tolerance = 0.05 * setpoint
    for i in range(len(y_sim)-1, -1, -1):
        if abs(y_sim[i] - setpoint) > tolerance:
            settling_time = tvec[i+1] if i < len(y_sim)-1 else tvec[-1]
            break
    else:
        settling_time = 0
    
    # 最大偏差
    max_deviation = np.max(np.abs(y_sim - setpoint))
    
    # 衰减比
    decay_ratio = 0
    peaks = []
    for i in range(1, len(y_sim)-1):
        if y_sim[i] > y_sim[i-1] and y_sim[i] > y_sim[i+1] and y_sim[i] > setpoint:
            peaks.append(y_sim[i])
    
    if len(peaks) >= 2:
        A1 = peaks[0] - setpoint
        A2 = peaks[1] - setpoint
        if A1 > 0:
            decay_ratio = A2 / A1
    
    return overshoot, max_deviation, settling_time, steady_error, decay_ratio

# 计算性能指标
metrics_initial = calculate_metrics(y_initial, t)
metrics_opt = calculate_metrics(y_opt, t)

# 打印性能指标
print('\n' + '='*65)
print('控制性能对比:')
print('='*65)
print(f"{'指标':<15} | {'初始PID':^15} | {'优化PID':^15} | {'改善率(%)':^15}")
print('-'*65)
print(f"{'超调量(℃)':<15} | {metrics_initial[0]:>15.2f} | {metrics_opt[0]:>15.2f} | "
      f"{(metrics_initial[0]-metrics_opt[0])/max(metrics_initial[0],0.01)*100:>14.1f}%")
print(f"{'最大偏差(℃)':<15} | {metrics_initial[1]:>15.2f} | {metrics_opt[1]:>15.2f} | "
      f"{(metrics_initial[1]-metrics_opt[1])/max(metrics_initial[1],0.01)*100:>14.1f}%")
print(f"{'5%回复时间(s)':<15} | {metrics_initial[2]:>15.2f} | {metrics_opt[2]:>15.2f} | "
      f"{(metrics_initial[2]-metrics_opt[2])/max(metrics_initial[2],0.01)*100:>14.1f}%")
print(f"{'余差(℃)':<15} | {metrics_initial[3]:>15.4f} | {metrics_opt[3]:>15.4f} | "
      f"{(metrics_initial[3]-metrics_opt[3])/max(abs(metrics_initial[3]),0.01)*100:>14.1f}%")
print(f"{'衰减比':<15} | {metrics_initial[4]:>15.2f} | {metrics_opt[4]:>15.2f} | "
      f"{(metrics_opt[4]-metrics_initial[4])/max(metrics_initial[4],0.01)*100:>14.1f}%")
print('='*65)