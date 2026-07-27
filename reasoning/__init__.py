"""CMI's core reasoning engine components (manuscript Sections 4-7, 10).

Sub-packages:
    reasoning.state    Cognitive Mission State sigma(m,t) (Section 4.1)
    reasoning.utility   Mission utility U(sigma) (Section 4.2)
    reasoning.ttrn      TTRN pipeline / Algorithm 1 (Section 5)
    reasoning.trust     Mission-aware GNN, Trust Propagation, Bayesian
                        posterior, Mission Criticality Model (Section 7)
    reasoning.causal    Causal Mission Intelligence interfaces (Section 10)
                        -- NOT YET IMPLEMENTED, see reasoning/causal/interfaces.py
"""
