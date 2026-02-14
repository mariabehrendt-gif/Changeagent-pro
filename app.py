"""
ChangeAgent Pro - Complete Single-File Version
Copy this entire code into app.py and deploy to Streamlit Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum
import uuid
from datetime import datetime
import plotly.express as px
import io

# PAGE SETUP
st.set_page_config(page_title="ChangeAgent Pro", page_icon="🎯", layout="wide")

# CSS STYLING
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f1f1f; }
    .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 20px; 
                 border-left: 5px solid #007acc; }
    .risk-red { border-left-color: #dc3545; background-color: #fff5f5; }
    .risk-amber { border-left-color: #ffc107; background-color: #fffbf0; }
    .risk-green { border-left-color: #28a745; background-color: #f0fff4; }
</style>
""", unsafe_allow_html=True)

# ==================== CORE CLASSES ====================

class RiskLevel(Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"

@dataclass
class Stakeholder:
    stakeholder_id: str
    email: str
    full_name: str
    role: str
    department: str
    location: str = "Office"
    tenure_years: float = 0.0
    digital_maturity: int = 3
    change_load: int = 0
    adkar_scores: Dict[str, float] = field(default_factory=dict)
    risk_level: RiskLevel = RiskLevel.GREEN
    barrier_point: str = ""
    overall_readiness: float = 0.0
    
    def calculate_readiness(self, weights: Dict[str, float]):
        if not self.adkar_scores:
            self.overall_readiness = 0.0
            return
        self.overall_readiness = sum(self.adkar_scores.get(k, 0) * v for k, v in weights.items())
        if self.overall_readiness < 60:
            self.risk_level = RiskLevel.RED
        elif self.overall_readiness < 75:
            self.risk_level = RiskLevel.AMBER
        else:
            self.risk_level = RiskLevel.GREEN
        if self.adkar_scores:
            self.barrier_point = min(self.adkar_scores, key=self.adkar_scores.get)

class ChangeAgent:
    def __init__(self):
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.intervention_queue: List[Dict] = []
        self.risk_hotspots: List[Dict] = []
        self.weights = {"awareness": 0.2, "desire": 0.25, "knowledge": 0.2, "ability": 0.25, "reinforcement": 0.1}
        self.interventions = {
            "awareness": {"red": ["🎯 Executive alignment: Refine burning platform narrative", "📢 Deploy C-suite video"], 
                         "amber": ["❓ FAQ amplification", "🍕 Lunch-and-learn sessions"]},
            "desire": {"red": ["🎁 WIIFM workshop: Role-specific benefits", "🏆 Champion activation"], 
                      "amber": ["📖 Success stories"]},
            "ability": {"red": ["🆘 Workload relief: Temporary backfill", "👨‍🏫 Intensive coaching"], 
                       "amber": ["📚 Micro-learning"]},
            "knowledge": {"red": ["🎓 Role-based bootcamp", "🔄 Process simulation"], 
                         "amber": ["📖 Self-paced path"]},
            "reinforcement": {"red": ["🔄 Sustainability plan", "🏅 Recognition program"], 
                           "amber": ["📈 Pulse checks"]}
        }
    
    def load_data(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            sid = str(uuid.uuid4())
            scores = {}
            for element in ["awareness", "desire", "knowledge", "ability", "reinforcement"]:
                cols = [c for c in df.columns if element in c.lower()]
                if cols:
                    vals = [float(row[c]) for c in cols if pd.notna(row[c]) and str(row[c]).replace('.','').isdigit()]
                    if vals:
                        scores[element] = (np.mean(vals) / 5) * 100 if max(vals) <= 5 else np.mean(vals)
            
            s = Stakeholder(
                stakeholder_id=sid,
                email=str(row.get("email", "")),
                full_name=str(row.get("full_name", row.get("name", "Unknown"))),
                role=str(row.get("role", "Staff")),
                department=str(row.get("department", "General")),
                location=str(row.get("location", "Office")),
                tenure_years=float(row.get("tenure_years", 0)) if pd.notna(row.get("tenure_years")) else 0,
                digital_maturity=int(row.get("digital_maturity", 3)) if pd.notna(row.get("digital_maturity")) else 3,
                change_load=int(row.get("change_load", 0)) if pd.notna(row.get("change_load")) else 0,
                adkar_scores=scores
            )
            s.calculate_readiness(self.weights)
            self.stakeholders[sid] = s
        
        self._analyze()
        return len(self.stakeholders)
    
    def _analyze(self):
        # Risk hotspots by department
        dept_stats = {}
        for s in self.stakeholders.values():
            d = s.department
            if d not in dept_stats:
                dept_stats[d] = {"total": 0, "red": 0, "scores": []}
            dept_stats[d]["total"] += 1
            dept_stats[d]["scores"].append(s.overall_readiness)
            if s.risk_level == RiskLevel.RED:
                dept_stats[d]["red"] += 1
        
        self.risk_hotspots = []
        for d, st in dept_stats.items():
            red_pct = (st["red"] / st["total"]) * 100 if st["total"] > 0 else 0
            avg_score = np.mean(st["scores"])
            if red_pct > 20 or avg_score < 65:
                self.risk_hotspots.append({
                    "department": d, "population": st["total"], "red_percentage": red_pct,
                    "average_readiness": avg_score, "severity": "critical" if red_pct > 40 else "high" if red_pct > 25 else "medium"
                })
        self.risk_hotspots.sort(key=lambda x: x["red_percentage"], reverse=True)
        
        # Generate interventions
        self.intervention_queue = []
        for s in self.stakeholders.values():
            if s.risk_level == RiskLevel.RED:
                actions = self.interventions.get(s.barrier_point, {}).get("red", ["General support"]).copy()
                if s.change_load > 3:
                    actions.insert(0, "🚨 URGENT: Change saturation")
                self.intervention_queue.append({
                    "priority": "P0-CRITICAL" if any(x in s.role.lower() for x in ["vp", "director", "manager", "head"]) else "P1-HIGH",
                    "target": s.full_name, "department": s.department, "barrier": s.barrier_point,
                    "score": s.overall_readiness, "actions": actions, "timeline": "24-48 hours"
                })
        
        # Sort interventions
        self.intervention_queue.sort(key=lambda x: 0 if x["priority"] == "P0-CRITICAL" else 1)
    
    def export_excel(self) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Stakeholders
            data = [{
                "name": s.full_name, "email": s.email, "department": s.department, "role": s.role,
                "readiness": round(s.overall_readiness, 1), "risk": s.risk_level.value, "barrier": s.barrier_point,
                **{k: round(v, 1) for k, v in s.adkar_scores.items()}
            } for s in self.stakeholders.values()]
            pd.DataFrame(data).to_excel(writer, sheet_name="Stakeholders", index=False)
            
            # Interventions
            if self.intervention_queue:
                pd.DataFrame(self.intervention_queue).to_excel(writer, sheet_name="Interventions", index=False)
            
            # Hotspots
            if self.risk_hotspots:
                pd.DataFrame(self.risk_hotspots).to_excel(writer, sheet_name="Hotspots", index=False)
        output.seek(0)
        return output.getvalue()

# ==================== UI ====================

# Session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'loaded' not in st.session_state:
    st.session_state.loaded = False

# Sidebar
st.sidebar.title("🎯 ChangeAgent Pro")
st.sidebar.markdown("**Zero-Touch Change Management**")
page = st.sidebar.radio("Navigate", ["📤 Upload", "📊 Dashboard", "👥 Stakeholders", "🎯 Interventions", "📥 Export"])

# MAIN PAGES
if page == "📤 Upload":
    st.markdown('<div class="main-header">📤 Upload Survey Data</div>', unsafe_allow_html=True)
    st.info("Required: email, full_name, role, department | Survey: awareness_q1, desire_q1, ability_q1, etc. (1-5 scale)")
    
    uploaded = st.file_uploader("Choose Excel or CSV", type=['xlsx', 'csv'])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        st.write(f"📊 Loaded {len(df)} rows")
        st.dataframe(df.head())
        
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Processing..."):
                agent = ChangeAgent()
                count = agent.load_data(df)
                st.session_state.agent = agent
                st.session_state.loaded = True
            st.success(f"✅ Analyzed {count} stakeholders!")
            st.balloons()

elif page == "📊 Dashboard":
    if not st.session_state.loaded:
        st.warning("⚠️ Upload data first")
    else:
        agent = st.session_state.agent
        st.markdown('<div class="main-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)
        
        total = len(agent.stakeholders)
        red = sum(1 for s in agent.stakeholders.values() if s.risk_level == RiskLevel.RED)
        amber = sum(1 for s in agent.stakeholders.values() if s.risk_level == RiskLevel.AMBER)
        green = total - red - amber
        avg = np.mean([s.overall_readiness for s in agent.stakeholders.values()])
        
        cols = st.columns(4)
        cols[0].metric("Stakeholders", total)
        cols[1].metric("Readiness", f"{avg:.1f}/100")
        cols[2].metric("Critical", red, f"{red/total*100:.1f}%" if total > 0 else "0%")
        cols[3].metric("Interventions", len(agent.intervention_queue))
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(names=["Critical", "At Risk", "Ready"], values=[red, amber, green],
                        title="Risk Distribution", color=["Critical", "At Risk", "Ready"],
                        color_discrete_map={"Critical": "#dc3545", "At Risk": "#ffc107", "Ready": "#28a745"})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            adkar = {
                "Element": ["Awareness", "Desire", "Knowledge", "Ability", "Reinforcement"],
                "Score": [np.mean([s.adkar_scores.get(e, 50) for s in agent.stakeholders.values()]) 
                         for e in ["awareness", "desire", "knowledge", "ability", "reinforcement"]]
            }
            fig2 = px.bar(adkar, x="Element", y="Score", title="ADKAR Health", range_y=[0, 100],
                         color="Score", color_continuous_scale=["red", "yellow", "green"])
            fig2.add_hline(y=60, line_dash="dash", line_color="red")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("🚨 Risk Hotspots")
        if agent.risk_hotspots:
            for h in agent.risk_hotspots[:3]:
                icon = "🔴" if h["severity"] == "critical" else "🟠"
                with st.expander(f"{icon} {h['department']}: {h['red_percentage']:.0f}% critical ({h['population']} people)"):
                    st.write(f"Avg Score: {h['average_readiness']:.1f}")
        else:
            st.success("✅ No critical hotspots")

elif page == "👥 Stakeholders":
    if not st.session_state.loaded:
        st.warning("⚠️ Upload data first")
    else:
        agent = st.session_state.agent
        st.markdown('<div class="main-header">👥 Stakeholder Analysis</div>', unsafe_allow_html=True)
        
        # Search
        search = st.text_input("Search by name")
        filtered = [(sid, s) for sid, s in agent.stakeholders.items() 
                   if not search or search.lower() in s.full_name.lower()]
        
        for sid, s in filtered:
            risk_class = f"risk-{s.risk_level.value}"
            with st.expander(f"{s.full_name} | {s.department} | {s.overall_readiness:.1f}"):
                st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Email:** {s.email}")
                c1.write(f"**Role:** {s.role}")
                c2.write(f"**Barrier:** {s.barrier_point}")
                c2.write(f"**Risk:** {s.risk_level.value.upper()}")
                c3.write("**ADKAR:**")
                for k, v in s.adkar_scores.items():
                    c3.write(f"{k}: {v:.1f}")
                st.markdown("</div>", unsafe_allow_html=True)

elif page == "🎯 Interventions":
    if not st.session_state.loaded:
        st.warning("⚠️ Upload data first")
    else:
        agent = st.session_state.agent
        st.markdown('<div class="main-header">🎯 Intervention Pipeline</div>', unsafe_allow_html=True)
        
        for i in agent.intervention_queue[:10]:
            icon = "🔴" if i["priority"] == "P0-CRITICAL" else "🟠"
            with st.expander(f"{icon} {i['priority']}: {i['target']} ({i['department']})"):
                st.write(f"**Barrier:** {i['barrier']} | **Score:** {i['score']:.1f}")
                st.write(f"**Timeline:** {i['timeline']}")
                st.write("**Actions:**")
                for a in i['actions']:
                    st.write(f"• {a}")

elif page == "📥 Export":
    if not st.session_state.loaded:
        st.warning("⚠️ Upload data first")
    else:
        agent = st.session_state.agent
        st.markdown('<div class="main-header">📥 Export Reports</div>', unsafe_allow_html=True)
        
        excel_data = agent.export_excel()
        st.download_button("📊 Download Excel Report", excel_data, "change_analysis.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("---")
st.sidebar.write("v1.0 | PROSCI Powered")
if st.session_state.loaded:
    st.sidebar.success(f"✅ {len(st.session_state.agent.stakeholders)} loaded")
