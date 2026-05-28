"""
violation_dashboard.py
Reusable Streamlit UI components for detailed PPE violation breakdown.
Used by both Ultra-Fast Video Analysis and Live Detection tabs.
"""

import streamlit as st
import io
import csv
from datetime import datetime
from collections import defaultdict

# Human-readable names for violation types
VIOLATION_LABEL_MAP = {
    'NO-Hardhat':          'Helmet Missing',
    'NO-Mask':             'Mask Missing',
    'NO-Safety Vest':      'Safety Jacket Missing',
    'NO-Gloves':           'Gloves Missing',
    'NO-Goggles':          'Goggles Missing',
    'NO-Boots':            'Boots Missing',
    'Missing Hardhat':     'Helmet Missing',
    'Missing Mask':        'Mask Missing',
    'Missing Safety Vest': 'Safety Jacket Missing',
    'Missing Gloves':      'Gloves Missing',
    'Missing Goggles':     'Goggles Missing',
    'Missing Boots':       'Boots Missing',
    'Missing Required PPE':'PPE Missing',
}

PPE_ICONS = {
    'Helmet Missing':         '⛑️',
    'Mask Missing':           '😷',
    'Safety Jacket Missing':  '🦺',
    'Gloves Missing':         '🧤',
    'Goggles Missing':        '🥽',
    'Boots Missing':          '🥾',
    'PPE Missing':            '⚠️',
    'Not Evaluated':          '⏸️',
}


def _friendly(vtype: str) -> str:
    return VIOLATION_LABEL_MAP.get(vtype, vtype)


def render_violation_breakdown(video_stats: dict, theme_config: dict = None):
    """
    Render a full PPE violation breakdown panel for Ultra-Fast Video Analysis.

    Parameters
    ----------
    video_stats : dict   Output from PPEDetectionEngine.process_video()
    theme_config : dict  Optional theme colours from theme_manager
    """
    tc = theme_config or {}
    accent   = tc.get('accent_color', '#667eea')
    danger   = tc.get('danger_color', '#ef4444')
    success  = tc.get('success_color', '#10b981')
    warning  = tc.get('warning_color', '#f59e0b')
    card_bg  = tc.get('card_bg', '#ffffff')
    text_pri = tc.get('text_primary', '#1f2937')
    text_sec = tc.get('text_secondary', '#6b7280')
    border   = tc.get('border_color', '#e5e7eb')

    ppe_counts   = video_stats.get('ppe_violation_counts', {})
    person_log   = video_stats.get('person_violation_log', [])
    frame_viols  = video_stats.get('frame_violations', [])
    total_viols  = video_stats.get('total_violations', 0)

    # ── Summary header ────────────────────────────────────────────────────────
    st.markdown("### 🔍 Detailed PPE Violation Breakdown")

    if not ppe_counts and not frame_viols:
        st.success("✅ No PPE violations detected in this video!")
        return

    # ── Per-PPE-type counts ───────────────────────────────────────────────────
    friendly_counts = defaultdict(int)
    for raw_type, count in ppe_counts.items():
        friendly_counts[_friendly(raw_type)] += count

    if friendly_counts:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border};
                    border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;
                    box-shadow:0 4px 20px rgba(0,0,0,0.06);">
            <h4 style="color:{text_pri}; margin:0 0 1rem 0;">
                📊 Violation Type Summary &nbsp;
                <span style="background:{danger}; color:white; padding:0.2rem 0.7rem;
                             border-radius:20px; font-size:0.85rem;">
                    Total: {total_viols}
                </span>
            </h4>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(min(len(friendly_counts), 4))
        for idx, (vtype, count) in enumerate(sorted(friendly_counts.items(),
                                                     key=lambda x: -x[1])):
            icon = PPE_ICONS.get(vtype, '⚠️')
            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,{danger}15,{danger}05);
                            border:1px solid {danger}40; border-radius:14px;
                            padding:1.2rem; text-align:center; margin-bottom:1rem;">
                    <div style="font-size:2rem;">{icon}</div>
                    <div style="font-size:1.8rem; font-weight:800; color:{danger};">
                        {count}
                    </div>
                    <div style="font-size:0.85rem; color:{text_sec}; font-weight:600;
                                text-transform:uppercase; letter-spacing:0.5px;">
                        {vtype}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Per-person violation log (first 20 unique person-frame combos) ────────
    if person_log:
        st.markdown(f"""
        <div style="background:{card_bg}; border:1px solid {border};
                    border-radius:16px; padding:1.5rem; margin-bottom:1.5rem;">
            <h4 style="color:{text_pri}; margin:0 0 1rem 0;">
                👷 Per-Person Violation Details
            </h4>
        </div>
        """, unsafe_allow_html=True)

        shown = 0
        for entry in person_log:
            if shown >= 20:
                st.caption(f"… and {len(person_log) - 20} more entries. Download CSV for full report.")
                break
            person_label = f"Worker {entry['person_idx'] + 1}" if entry['person_idx'] >= 0 else "Worker"
            viols_friendly = [_friendly(v) for v in entry['violations']]
            viol_tags = " &nbsp; ".join(
                f'<span style="background:{danger}; color:white; padding:0.15rem 0.5rem; '
                f'border-radius:8px; font-size:0.8rem;">'
                f'{PPE_ICONS.get(v,"⚠️")} {v}</span>'
                for v in viols_friendly
            )
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,{danger}08,{card_bg});
                        border-left:4px solid {danger}; border-radius:8px;
                        padding:0.8rem 1rem; margin-bottom:0.5rem;
                        display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <strong style="color:{text_pri};">👷 {person_label}</strong>
                    &nbsp;&nbsp; {viol_tags}
                </div>
                <div style="color:{text_sec}; font-size:0.8rem; white-space:nowrap;">
                    🎬 Frame {entry['frame']} &nbsp; ⏱️ {entry['timestamp']}s
                </div>
            </div>
            """, unsafe_allow_html=True)
            shown += 1

    # ── Compliant frames summary ──────────────────────────────────────────────
    total_frames = video_stats.get('processed_frames', 1)
    violation_frames = len(frame_viols)
    compliant_frames = max(0, total_frames - violation_frames)
    compliance_pct = (compliant_frames / total_frames * 100) if total_frames > 0 else 100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{success}15,{success}05);
                    border:1px solid {success}40; border-radius:14px;
                    padding:1.2rem; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:{success};">
                {compliant_frames:,}
            </div>
            <div style="font-size:0.85rem; color:{text_sec}; font-weight:600;">
                ✅ Compliant Frames
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{danger}15,{danger}05);
                    border:1px solid {danger}40; border-radius:14px;
                    padding:1.2rem; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:{danger};">
                {violation_frames:,}
            </div>
            <div style="font-size:0.85rem; color:{text_sec}; font-weight:600;">
                ❌ Violation Frames
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        color = success if compliance_pct >= 80 else warning if compliance_pct >= 60 else danger
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{color}15,{color}05);
                    border:1px solid {color}40; border-radius:14px;
                    padding:1.2rem; text-align:center;">
            <div style="font-size:1.8rem; font-weight:800; color:{color};">
                {compliance_pct:.1f}%
            </div>
            <div style="font-size:0.85rem; color:{text_sec}; font-weight:600;">
                📊 Frame Compliance
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📥 Download Violation Report")

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(['Frame', 'Timestamp_s', 'Person', 'Violation', 'Generated'])
    ts_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for entry in person_log:
        person_label = f"Worker {entry['person_idx']+1}" if entry['person_idx'] >= 0 else "Worker"
        for v in entry['violations']:
            writer.writerow([entry['frame'], entry['timestamp'],
                             person_label, _friendly(v), ts_now])

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📥 Download Violations CSV",
            csv_buf.getvalue(),
            file_name=f"ppe_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        # Summary text report
        lines = [
            "PPE Violation Report",
            f"Generated: {ts_now}",
            "=" * 40,
            f"Total Violations: {total_viols}",
            ""
        ]
        for vtype, cnt in sorted(friendly_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {PPE_ICONS.get(vtype,'?')} {vtype}: {cnt}")
        lines += ["", "Per-Person Details:", "-" * 30]
        for entry in person_log:
            person_label = f"Worker {entry['person_idx']+1}" if entry['person_idx'] >= 0 else "Worker"
            viols = ", ".join(_friendly(v) for v in entry['violations'])
            lines.append(f"Frame {entry['frame']} ({entry['timestamp']}s) | {person_label}: {viols}")

        st.download_button(
            "📄 Download Summary TXT",
            "\n".join(lines),
            file_name=f"ppe_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def render_live_violation_panel(compliance_stats: dict, frame_count: int = 0,
                                 theme_config: dict = None):
    """
    Render a compact real-time violation panel for Live Detection tab.
    Shows violations, compliant status, and not-evaluated (not visible) items.
    """
    tc = theme_config or {}
    danger   = tc.get('danger_color', '#ef4444')
    success  = tc.get('success_color', '#10b981')
    card_bg  = tc.get('card_bg', '#ffffff')
    text_pri = tc.get('text_primary', '#1f2937')
    text_sec = tc.get('text_secondary', '#6b7280')
    border   = tc.get('border_color', '#e5e7eb')

    violations   = compliance_stats.get('violations', [])
    total_people = compliance_stats.get('total_people', 0)

    # Separate real violations from skipped-only entries
    real_viols = [v for v in violations if not v.get('skipped_only', False)]

    if not real_viols:
        msg = (f"All {total_people} worker(s) fully compliant!"
               if total_people > 0 else "No Violations Detected")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{success}15,{success}05);
                    border:1px solid {success}40; border-radius:12px;
                    padding:1rem; text-align:center;">
            <strong style="color:{success};">\u2705 {msg}</strong>
        </div>
        """, unsafe_allow_html=True)
        return

    # Group violations and not-evaluated items by person
    person_map = defaultdict(list)
    not_eval_map = defaultdict(list)
    for v in violations:
        pidx = v.get('associated_person_idx', -1)
        if not v.get('skipped_only', False):
            person_map[pidx].append(_friendly(v.get('type', 'Unknown')))
        for ne in v.get('not_evaluated', []):
            not_eval_map[pidx].append(ne)

    real_count = len(real_viols)
    st.markdown(f"""
    <div style="background:{card_bg}; border:1px solid {border};
                border-radius:12px; padding:1rem; margin-bottom:1rem;">
        <h5 style="color:{text_pri}; margin:0 0 0.8rem 0;">
            \u26a0\ufe0f Live Violations &nbsp;
            <span style="background:{danger}; color:white; padding:0.1rem 0.5rem;
                         border-radius:12px; font-size:0.8rem;">
                {real_count} issue(s)
            </span>
        </h5>
    </div>
    """, unsafe_allow_html=True)

    all_pidx = sorted(set(list(person_map.keys()) + list(not_eval_map.keys())))
    for pidx in all_pidx:
        person_label = f"Worker {pidx + 1}" if pidx >= 0 else "Worker"
        viols     = person_map.get(pidx, [])
        not_evals = list(set(not_eval_map.get(pidx, [])))

        viol_html = " &nbsp; ".join(
            f'<span style="background:{danger}20; color:{danger}; border:1px solid {danger}40; '
            f'padding:0.1rem 0.4rem; border-radius:6px; font-size:0.8rem;">'
            f'{PPE_ICONS.get(v, "\u26a0\ufe0f")} {v}</span>'
            for v in viols
        )
        ne_html = " &nbsp; ".join(
            f'<span style="color:#94a3b8; font-size:0.75rem;">'
            f'\u23f8\ufe0f {ne} (not visible)</span>'
            for ne in not_evals
        ) if not_evals else ""

        border_color = danger if viols else success
        ne_section = f'<br><small>{ne_html}</small>' if ne_html else ""
        st.markdown(f"""
        <div style="border-left:3px solid {border_color}; padding:0.5rem 0.8rem;
                    margin-bottom:0.4rem; background:{border_color}05;
                    border-radius:0 8px 8px 0;">
            <strong style="color:{text_pri};">\U0001f477 {person_label}</strong>
            &nbsp;&nbsp; {viol_html}{ne_section}
        </div>
        """, unsafe_allow_html=True)

