# Gradio UI Quarantine Segregation and Diagnostic Cards

Rendering all generated pitches in a single flat Markdown list obscured whether a given brief passed all investment rubrics or was quarantined due to unresolved critique issues. We decided to redesign the pitch display in `app.py` into two distinct sections: '🏆 Approved Pitch Briefs' (certified investor-ready) and '⚠️ Quarantined Pitch Briefs' (with attached Critic Diagnostic Reports displaying failing checks and revision audit trails). We also added dedicated download buttons for each category.
