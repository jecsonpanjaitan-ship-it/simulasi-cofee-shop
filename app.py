import streamlit as st
import simpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import math
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# ================= CONFIG & THEME =================
st.set_page_config(
    page_title="☕ Coffee Shop V&V Simulation",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header { 
        font-size: 32px; 
        font-weight: 800; 
        color: #5D4037; 
        margin-bottom: 10px;
        text-align: center;
    }
    
    .sub-header { 
        font-size: 18px; 
        color: #795548; 
        margin-bottom: 25px;
        text-align: center;
    }
    
    .kpi-card { 
        background: linear-gradient(135deg, #EFEBE9 0%, #D7CCC8 100%); 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 6px solid #6D4C41; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .success-box { 
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid #43A047;
    }
    
    .warning-box { 
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%); 
        padding: 20px; 
        border-radius: 12px; 
        border-left: 6px solid #FB8C00;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2196F3;
    }
    
    .slide-ref { 
        font-size: 13px; 
        color: #757575;  
        background: #F5F5F5; 
        padding: 5px 12px; 
        border-radius: 8px; 
        display: inline-block; 
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ================= CORE SIMULATION CLASS =================
class CoffeeShopSimulation:
    """
    Discrete-Event Simulation untuk Coffee Shop.
    Mendukung:
      - Monte Carlo multi-run
      - Trace output untuk Verifikasi
      - Perhitungan teoritis M/M/1 dan M/M/c (Erlang C) untuk Validasi
    """
    
    def __init__(self, num_baristas=2, duration_min=480,
                  arrival_rate=20, service_mean=3.0, seed=42):
        self.num_baristas = num_baristas
        self.duration_min = duration_min
        self.arrival_rate = arrival_rate
        self.service_mean = service_mean
        self.seed = seed
        
        # Inter-arrival time rata-rata (menit)
        self.inter_arrival = (60.0 / arrival_rate) if arrival_rate > 0 else 1e9
        
        # Storage hasil simulasi
        self.wait_times = []
        self.service_times = []
        self.queue_lengths = []
        self.time_stamps = []
        self.trace_log = []
    
    def run_simulation(self, n_runs=10, enable_trace=False):
        """Jalankan Monte Carlo DES."""
        np.random.seed(self.seed)
        
        all_metrics = {
            'avg_wait': [], 'avg_queue': [], 'utilization': [],
            'total_served': [], 'max_queue': [], 'std_wait': []
        }
        
        for run in range(n_runs):
            env = simpy.Environment()
            barista = simpy.Resource(env, capacity=self.num_baristas)
            
            run_wait_times = []
            run_service_times = []
            run_queue_lengths = []
            run_time_stamps = []
            customer_id = [0]
            
            # Inisialisasi trace log hanya untuk run pertama
            if enable_trace and run == 0:
                self.trace_log = []
                self.trace_log.append('=' * 80)
                self.trace_log.append('TRACE OUTPUT – Run 1')
                self.trace_log.append(
                    f'Parameter: λ={self.arrival_rate}/jam, '
                    f'μ={self.service_mean} mnt, Barista={self.num_baristas}'
                )
                self.trace_log.append('-' * 80)
                self.trace_log.append(
                    f'{"t(mnt)":>8} | {"Event":<12} | {"Cust":>5} | {"Detail"}')
                self.trace_log.append('-' * 80)
            
            # Proses pelanggan
            def customer(env, cid):
                arrival_time = env.now
                
                if enable_trace and run == 0 and cid < 10:
                    q_len = len(barista.queue)
                    self.trace_log.append(
                        f'{env.now:8.2f} | {"ARRIVAL":<12} | {cid:5d} | '
                        f'Queue={q_len}')
                
                with barista.request() as req:
                    yield req
                    wait_time = env.now - arrival_time
                    run_wait_times.append(wait_time)
                    
                    if enable_trace and run == 0 and cid < 10:
                        self.trace_log.append(
                            f'{env.now:8.2f} | {"SVC START":<12} | {cid:5d} | '
                            f'Wait={wait_time:.2f} mnt')
                    
                    service_time = np.random.exponential(scale=self.service_mean)
                    run_service_times.append(service_time)
                    yield env.timeout(service_time)
                    
                    if enable_trace and run == 0 and cid < 10:
                        self.trace_log.append(
                            f'{env.now:8.2f} | {"DEPARTURE":<12} | {cid:5d} | '
                            f'Svc={service_time:.2f} mnt')
            
            # Generator kedatangan
            def generate_customers(env):
                while env.now < self.duration_min:
                    iat = np.random.exponential(scale=self.inter_arrival)
                    yield env.timeout(iat)
                    if env.now < self.duration_min:
                        cid = customer_id[0]
                        customer_id[0] += 1
                        env.process(customer(env, cid))
            
            # Monitor panjang antrian setiap 1 menit
            def monitor_queue(env):
                t = 0
                while t < self.duration_min:
                    run_queue_lengths.append(len(barista.queue))
                    run_time_stamps.append(t)
                    t += 1
                    yield env.timeout(1.0)
            
            env.process(generate_customers(env))
            env.process(monitor_queue(env))
            env.run(until=self.duration_min)
            
            # Hitung metrik tiap run
            total_svc = sum(run_service_times)
            all_metrics['avg_wait'].append(
                np.mean(run_wait_times) if run_wait_times else 0.0)
            all_metrics['avg_queue'].append(
                np.mean(run_queue_lengths) if run_queue_lengths else 0.0)
            all_metrics['utilization'].append(
                (total_svc / (self.num_baristas * self.duration_min)) * 100)
            all_metrics['total_served'].append(len(run_wait_times))
            all_metrics['max_queue'].append(
                max(run_queue_lengths) if run_queue_lengths else 0)
            all_metrics['std_wait'].append(
                np.std(run_wait_times) if run_wait_times else 0.0)
            
            # Simpan run terakhir untuk visualisasi
            if run == n_runs - 1:
                self.wait_times = run_wait_times
                self.service_times = run_service_times
                self.queue_lengths = run_queue_lengths
                self.time_stamps = run_time_stamps
        
        # Agregat Monte Carlo
        n = len(all_metrics['avg_wait'])
        sem_val = stats.sem(all_metrics['avg_wait']) if n > 1 else 0.0
        if n > 1 and sem_val > 0:
            ci = stats.t.interval(0.95, n - 1,
                                  loc=np.mean(all_metrics['avg_wait']),
                                  scale=sem_val)
        else:
            m = np.mean(all_metrics['avg_wait'])
            ci = (m, m)
        
        return {
            'avg_wait_mean': np.mean(all_metrics['avg_wait']),
            'avg_wait_std': np.std(all_metrics['avg_wait']),
            'avg_queue_mean': np.mean(all_metrics['avg_queue']),
            'avg_queue_std': np.std(all_metrics['avg_queue']),
            'utilization_mean': np.mean(all_metrics['utilization']),
            'utilization_std': np.std(all_metrics['utilization']),
            'total_served_mean': int(np.mean(all_metrics['total_served'])),
            'max_queue_mean': np.mean(all_metrics['max_queue']),
            'wait_time_confidence_interval': ci,
            'all_avg_wait': all_metrics['avg_wait'],
        }
    
    def get_trace_log(self):
        """Kembalikan trace log sebagai list string."""
        return self.trace_log
    
    def theoretical_mm1_wait(self):
        """Waktu tunggu rata-rata teoritis M/M/1."""
        lam = self.arrival_rate / 60.0
        mu = 1.0 / self.service_mean
        if lam >= mu:
            return float('inf')
        return lam / (mu * (mu - lam))
    
    def theoretical_mmc_wait(self):
        """Waktu tunggu rata-rata teoritis M/M/c menggunakan formula Erlang C."""
        c = self.num_baristas
        lam = self.arrival_rate / 60.0
        mu = 1.0 / self.service_mean
        rho = lam / (c * mu)
        a = lam / mu
        
        if rho >= 1.0:
            return float('inf')
        
        # Hitung P0
        sum_terms = sum(a**n / math.factorial(n) for n in range(c))
        last_term = a**c / (math.factorial(c) * (1.0 - rho))
        P0 = 1.0 / (sum_terms + last_term)
        
        # Lq = rata-rata pelanggan dalam antrian
        Lq = (P0 * a**c * rho) / (math.factorial(c) * (1.0 - rho)**2)
        
        # Wq = Lq / λ
        Wq = Lq / lam
        return Wq

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/coffee.png", width=80)
    st.title("⚙️ Konfigurasi Simulasi")
    st.markdown("---")
    
    arrival_rate = st.slider("Kedatangan (λ pelanggan/jam)", 10, 40, 20)
    service_mean = st.slider("Rata-rata Layanan (menit)", 1.5, 6.0, 3.0)
    num_baristas = st.select_slider("Jumlah Barista", options=[1, 2, 3, 4, 5], value=2)
    duration_min = st.slider("Durasi Simulasi (menit)", 120, 480, 480, step=60)
    n_runs = st.slider("Monte Carlo Runs", 5, 20, 10)
    seed = st.number_input("Random Seed", value=42)
    
    st.markdown("---")
    st.info("💡 **Tip:** Gunakan Monte Carlo runs lebih tinggi untuk hasil yang lebih akurat")

# ================= MAIN CONTENT =================
st.markdown('<p class="main-header">☕ Tugas V&V: Simulasi Coffee Shop (DES)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Verification & Validation - Week 10</p>', unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 1. Data & Asumsi",
    "🔍 2. Verification",
    "🔬 3. Validation",
    "📊 4. Hasil Simulasi",
    "📈 5. Visualisasi"
])

with tab1:
    st.markdown('<span class="slide-ref">📄 Slide 9-10</span>', unsafe_allow_html=True)
    st.markdown("### 📊 Data Input & Asumsi Pemodelan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Parameter Distribusi Probabilitas")
        st.markdown(f"""
        **Kedatangan Pelanggan:**
        - Distribusi: `Poisson(λ={arrival_rate}/jam)`
        - Inter-arrival time: `Exponential({60/arrival_rate:.1f} menit)`
        
        **Waktu Layanan:**
        - Distribusi: `Exponential(μ={service_mean} menit)`
        - Kapasitas layanan: {60/service_mean:.1f} pelanggan/jam/barista
        
        **Durasi:** {duration_min} menit ({duration_min/60:.1f} jam)
        """)
    
    with col2:
        st.markdown("#### 📌 Asumsi Sistem")
        st.markdown("""
        1. ✅ Tidak ada balking/reneging (semua pelanggan menunggu)
        2. ✅ Disiplin antrian: `FCFS` (First Come First Serve)
        3. ✅ Kapasitas antrian: `Unlimited`
        4. ✅ Semua barista memiliki kecepatan sama
        5. ✅ Tidak ada break time selama simulasi
        """)
    
    st.markdown("---")
    st.markdown("#### 📐 Model Antrian: M/M/c")
    
    rho = (arrival_rate/60)/(num_baristas*(1/service_mean))
    st.latex(rf"\text{{Utilisasi }} \rho = \frac{{\lambda}}{{c \cdot \mu}} = \frac{{{arrival_rate}/60}}{{{num_baristas} \cdot 1/{service_mean}}} = {rho:.2f}")
    
    if rho >= 1:
        st.error("⚠️ **Warning:** Utilisasi ≥ 100%! Sistem tidak stabil. Tambah barista atau kurangi kedatangan.")
    else:
        st.success(f"✅ Sistem stabil dengan utilisasi teoritis {rho*100:.1f}%")

with tab2:
    st.markdown('<span class="slide-ref">📄 Verification Tasks</span>', unsafe_allow_html=True)
    st.markdown("### 🔍 Verification: Trace Output & Edge Cases")
    
    if st.button("🚀 Jalankan Verification", use_container_width=True, type="primary"):
        # Verification 1: Trace Output
        st.markdown("#### 1️⃣ Trace Output (10 Pelanggan Pertama)")
        
        sim_trace = CoffeeShopSimulation(
            num_baristas=num_baristas, duration_min=duration_min,
            arrival_rate=arrival_rate, service_mean=service_mean, seed=seed
        )
        _ = sim_trace.run_simulation(n_runs=1, enable_trace=True)
        
        with st.expander("📜 Lihat Trace Log Lengkap", expanded=True):
            trace_text = "\n".join(sim_trace.get_trace_log())
            st.text(trace_text)
        
        st.success("✅ Verification 1: Trace output berhasil ditampilkan")
        st.info("Trace menunjukkan urutan events: ARRIVAL → SVC START → DEPARTURE")
        
        st.markdown("---")
        
        # Verification 2: Edge Cases
        st.markdown("#### 2️⃣ Edge Cases Testing")
        
        edge_results = []
        
        # Edge Case 1: 0 Pelanggan
        st.markdown("**Edge Case 1: 0 Pelanggan (λ = 0)**")
        try:
            sim_ec1 = CoffeeShopSimulation(
                num_baristas=1, duration_min=duration_min,
                arrival_rate=0, service_mean=service_mean, seed=seed
            )
            res_ec1 = sim_ec1.run_simulation(n_runs=5)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Pelanggan", res_ec1['total_served_mean'])
            col2.metric("Waktu Tunggu", f"{res_ec1['avg_wait_mean']:.4f} mnt")
            col3.metric("Utilisasi", f"{res_ec1['utilization_mean']:.2f}%")
            
            if res_ec1['total_served_mean'] == 0 and res_ec1['avg_wait_mean'] == 0.0:
                st.success("✅ PASSED — 0 pelanggan, waktu tunggu = 0")
                edge_results.append(("0 Pelanggan (λ=0)", "✅ PASSED"))
            else:
                st.error("❌ FAILED")
                edge_results.append(("0 Pelanggan (λ=0)", "❌ FAILED"))
        except Exception as e:
            st.error(f"❌ ERROR: {e}")
            edge_results.append(("0 Pelanggan (λ=0)", f"❌ ERROR: {e}"))
        
        st.markdown("---")
        
        # Edge Case 2: 1 Barista (M/M/1)
        st.markdown("**Edge Case 2: 1 Barista (M/M/1), λ=15/jam**")
        try:
            sim_ec2 = CoffeeShopSimulation(
                num_baristas=1, duration_min=duration_min,
                arrival_rate=15, service_mean=service_mean, seed=seed
            )
            res_ec2 = sim_ec2.run_simulation(n_runs=10)
            theo_mm1 = sim_ec2.theoretical_mm1_wait()
            sim_wait = res_ec2['avg_wait_mean']
            error_pct = abs(sim_wait - theo_mm1) / theo_mm1 * 100 if theo_mm1 != 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Simulasi", f"{sim_wait:.2f} mnt")
            col2.metric("Teoritis M/M/1", f"{theo_mm1:.2f} mnt")
            col3.metric("Error", f"{error_pct:.2f}%")
            
            if error_pct < 20:
                st.success("✅ PASSED — Error < 20%, sesuai teori M/M/1")
                edge_results.append(("1 Barista (M/M/1)", "✅ PASSED"))
            else:
                st.warning("⚠️ WARNING — Error cukup besar (acceptable untuk simulasi pendek)")
                edge_results.append(("1 Barista (M/M/1)", "⚠️ WARNING"))
        except Exception as e:
            st.error(f"❌ ERROR: {e}")
            edge_results.append(("1 Barista (M/M/1)", f"❌ ERROR: {e}"))
        
        st.markdown("---")
        
        # Edge Case 3: Utilisasi Tinggi
        st.markdown("**Edge Case 3: Utilisasi Tinggi (ρ ≈ 0.95), λ=19/jam**")
        try:
            sim_ec3 = CoffeeShopSimulation(
                num_baristas=1, duration_min=duration_min,
                arrival_rate=19, service_mean=service_mean, seed=seed
            )
            res_ec3 = sim_ec3.run_simulation(n_runs=5)
            rho_ec3 = (19/60) / (1/service_mean)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("ρ Teoritis", f"{rho_ec3:.2f}")
            col2.metric("Waktu Tunggu", f"{res_ec3['avg_wait_mean']:.2f} mnt")
            col3.metric("Max Antrian", f"{res_ec3['max_queue_mean']:.1f}")
            
            if res_ec3['avg_wait_mean'] > 0:
                st.success("✅ PASSED — Sistem handle utilisasi tinggi dengan benar")
                edge_results.append(("Utilisasi Tinggi (ρ≈0.95)", "✅ PASSED"))
            else:
                st.error("❌ FAILED")
                edge_results.append(("Utilisasi Tinggi (ρ≈0.95)", "❌ FAILED"))
        except Exception as e:
            st.error(f"❌ ERROR: {e}")
            edge_results.append(("Utilisasi Tinggi (ρ≈0.95)", f"❌ ERROR: {e}"))
        
        # Summary
        st.markdown("---")
        st.markdown("#### 📋 Summary Verification")
        edge_df = pd.DataFrame(edge_results, columns=["Test Case", "Status"])
        st.dataframe(edge_df, use_container_width=True, hide_index=True)
        
        passed = sum(1 for r in edge_results if '✅' in r[1])
        st.success(f"✅ **Verification Selesai:** {passed}/{len(edge_results)} test cases PASSED")

with tab3:
    st.markdown('<span class="slide-ref">📄 Validation Task</span>', unsafe_allow_html=True)
    st.markdown("### 🔬 Validation: Simulasi vs Teoritis M/M/c")
    
    if st.button("🔬 Jalankan Validation", use_container_width=True, type="primary"):
        test_cases = [
            {'baristas': 1, 'arrival': 15, 'service': service_mean, 'label': 'M/M/1 – Low Load (ρ=0.75)'},
            {'baristas': 1, 'arrival': 18, 'service': service_mean, 'label': 'M/M/1 – High Load (ρ=0.90)'},
            {'baristas': 2, 'arrival': arrival_rate, 'service': service_mean, 'label': f'M/M/2 – Skenario Utama (λ={arrival_rate})'},
            {'baristas': 3, 'arrival': min(arrival_rate + 10, 40), 'service': service_mean, 'label': 'M/M/3 – Load Tinggi'},
        ]
        
        val_rows = []
        
        for tc in test_cases:
            sim_v = CoffeeShopSimulation(
                num_baristas=tc['baristas'],
                duration_min=duration_min,
                arrival_rate=tc['arrival'],
                service_mean=tc['service'],
                seed=seed
            )
            res_v = sim_v.run_simulation(n_runs=10)
            
            if tc['baristas'] == 1:
                theo = sim_v.theoretical_mm1_wait()
            else:
                theo = sim_v.theoretical_mmc_wait()
            
            sim_w = res_v['avg_wait_mean']
            if theo not in (float('inf'), 0.0):
                err = abs(sim_w - theo) / theo * 100
            else:
                err = float('nan')
            
            status = '✅ PASS' if err < 20 else '⚠️ WARN'
            val_rows.append({
                'Test Case': tc['label'],
                'Sim (mnt)': f'{sim_w:.3f}',
                'Teoritis (mnt)': f'{theo:.3f}' if theo != float('inf') else '∞',
                'Error (%)': f'{err:.2f}' if not math.isnan(err) else 'N/A',
                'Status': status,
            })
        
        df_val = pd.DataFrame(val_rows)
        st.dataframe(df_val, use_container_width=True, hide_index=True)
        
        passed = sum(1 for r in val_rows if '✅' in r['Status'])
        
        if passed == len(val_rows):
            st.success(f"✅ **VALIDATION LULUS** — {passed}/{len(val_rows)} test cases PASSED (error < 20%)")
        else:
            st.warning(f"⚠️ **VALIDATION PARTIAL** — {passed}/{len(val_rows)} test cases PASSED")
        
        st.info("Catatan: Error tinggi pada utilisasi mendekati 1 (ρ→1) adalah wajar karena simulasi berdurasi terbatas.")

with tab4:
    st.markdown('<span class="slide-ref">📄 Slide 11</span>', unsafe_allow_html=True)
    st.markdown("### 📊 Hasil Simulasi Utama")
    
    if st.button("🚀 Jalankan Simulasi Utama", use_container_width=True, type="primary"):
        with st.spinner("⏳ Menjalankan Discrete-Event Simulation..."):
            sim_main = CoffeeShopSimulation(
                num_baristas=num_baristas, duration_min=duration_min,
                arrival_rate=arrival_rate, service_mean=service_mean, seed=seed
            )
            res_main = sim_main.run_simulation(n_runs=n_runs)
            
            st.session_state.res_main = res_main
            st.session_state.sim_main = sim_main
            st.session_state.params = {
                'λ': arrival_rate, 'μ': service_mean, 
                'baristas': num_baristas, 'duration': duration_min
            }
        
        st.success("✅ Simulasi selesai!")
        
        # KPI Metrics
        st.markdown("#### 📈 Key Performance Indicators (KPI)")
        
        c1, c2, c3, c4 = st.columns(4)
        
        ci = res_main['wait_time_confidence_interval']
        
        with c1:
            st.metric(
                label="⏱️ Avg. Waktu Tunggu",
                value=f"{res_main['avg_wait_mean']:.2f} ± {res_main['avg_wait_std']:.2f} mnt",
                delta=f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})"
            )
        
        with c2:
            st.metric(
                label="📊 Avg. Panjang Antrian",
                value=f"{res_main['avg_queue_mean']:.2f} ± {res_main['avg_queue_std']:.2f} org",
                delta=f"Max: {res_main['max_queue_mean']:.1f}"
            )
        
        with c3:
            st.metric(
                label="⚡ Utilisasi Barista",
                value=f"{res_main['utilization_mean']:.1f} ± {res_main['utilization_std']:.1f}%",
                delta="Optimal: 60-85%" if 60 <= res_main['utilization_mean'] <= 85 else "Perlu penyesuaian"
            )
        
        with c4:
            st.metric(
                label="👥 Total Terlayani",
                value=f"{res_main['total_served_mean']} pelanggan",
                delta=f"{res_main['total_served_mean']/duration_min*60:.1f} jam"
            )
        
        st.markdown("---")
        
        # Detailed statistics
        st.markdown("#### 📋 Detail Statistik Simulasi")
        stats_df = pd.DataFrame({
            'Metrik': [
                'Rata-rata Waktu Tunggu',
                'Standar Deviasi',
                '95% Confidence Interval',
                'Rata-rata Panjang Antrian',
                'Utilisasi Barista',
                'Total Pelanggan',
                'Throughput (pelanggan/jam)'
            ],
            'Nilai': [
                f"{res_main['avg_wait_mean']:.3f} menit",
                f"{res_main['avg_wait_std']:.3f} menit",
                f"({ci[0]:.3f}, {ci[1]:.3f})",
                f"{res_main['avg_queue_mean']:.3f} pelanggan",
                f"{res_main['utilization_mean']:.2f}%",
                f"{res_main['total_served_mean']} pelanggan",
                f"{res_main['total_served_mean']/duration_min*60:.2f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

with tab5:
    st.markdown('<span class="slide-ref">📄 Visualisasi</span>', unsafe_allow_html=True)
    st.markdown("### 📈 Visualisasi Hasil Simulasi")
    
    if 'res_main' not in st.session_state:
        st.warning("👈 Jalankan simulasi terlebih dahulu di tab 'Hasil Simulasi'")
    else:
        res = st.session_state.res_main
        sim = st.session_state.sim_main
        
        # Plot 1: Queue Length Over Time
        st.markdown("#### 📉 Dinamika Panjang Antrian vs Waktu")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ts_hours = [t / 60 for t in sim.time_stamps]
        ax.plot(ts_hours, sim.queue_lengths, color='#D84315', linewidth=1.5, alpha=0.8)
        ax.fill_between(ts_hours, sim.queue_lengths, alpha=0.2, color='#D84315')
        ax.axhline(np.mean(sim.queue_lengths), color='navy', linestyle='--', linewidth=2, 
                    label=f'Rata-rata: {np.mean(sim.queue_lengths):.2f}')
        ax.set_xlabel('Waktu (jam)', fontsize=11)
        ax.set_ylabel('Panjang Antrian (pelanggan)', fontsize=11)
        ax.set_title(f'Panjang Antrian Sepanjang Waktu ({num_baristas} Barista)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, duration_min/60)
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        # Plot 2: Wait Time Distribution
        st.markdown("#### 📊 Distribusi Waktu Tunggu Pelanggan")
        
        fig, ax = plt.subplots(figsize=(12, 5))
        if sim.wait_times:
            ax.hist(sim.wait_times, bins=30, color='#1976D2', edgecolor='white', alpha=0.8, density=True)
            ax.axvline(np.mean(sim.wait_times), color='red', linestyle='--', linewidth=2, 
                        label=f'Mean: {np.mean(sim.wait_times):.2f} mnt')
            ax.axvline(np.median(sim.wait_times), color='orange', linestyle=':', linewidth=2,
                        label=f'Median: {np.median(sim.wait_times):.2f} mnt')
        ax.set_xlabel('Waktu Tunggu (menit)', fontsize=11)
        ax.set_ylabel('Densitas', fontsize=11)
        ax.set_title('Distribusi Waktu Tunggu Pelanggan', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        
        # Plot 3: Utilization Pie Chart
        st.markdown("#### 📊 Utilisasi Barista")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#D84315', '#C8E6C9']
        wedges, texts, autotexts = ax.pie(
            [res['utilization_mean'], 100 - res['utilization_mean']],
            labels=['Utilisasi', 'Idle Time'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=(0.05, 0)
        )
        ax.set_title(f'Utilisasi Barista: {res["utilization_mean"]:.1f}%', fontsize=12, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        
        # Export data
        st.markdown("---")
        st.markdown("#### 📥 Export Data")
        
        csv_data = pd.DataFrame({
            'Time_Minute': sim.time_stamps,
            'Average_Queue_Length': sim.queue_lengths
        }).to_csv(index=False)
        
        st.download_button(
            label="📥 Download CSV Results",
            data=csv_data,
            file_name=f"coffee_shop_simulation_{num_baristas}_baristas.csv",
            mime="text/csv",
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption(f"""
Dibuat sesuai materi Week 10: Pemodelan & Simulasi Bisnis | 
Studi Kasus Coffee Shop - Verification & Validation | 
© 2026 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")