"""Default GAIA research assumptions; none are official Wharton parameters."""

DEFAULT_GAIA = {
    "sector_weights": {
        "growth": 0.25,
        "earnings_revisions": 0.20,
        "profitability": 0.20,
        "valuation": 0.15,
        "risk_adjusted_momentum": 0.20,
    },
    "gfss_weights": {
        "traditional_quant": 0.20,
        "fundamental": 0.20,
        "narrative_delta": 0.10,
        "catalyst": 0.10,
        "management": 0.10,
        "ai_consensus": 0.15,
        "risk": 0.15,
    },
    "agents": [
        "Fundamental Analyst",
        "Risk Analyst",
        "Macro Analyst",
        "Devil's Advocate",
        "Client-Liability Analyst",
    ],
    "uncertainty": {
        "min_ai_confidence": 0.60,
        "max_ads": 0.75,
        "require_human_approval": True,
    },
    "allocation": {
        "stock_sleeve": 0.40,
        "min_stock_weight": 0.02,
        "max_stock_weight": 0.06,
        "sector_cap_within_stock_sleeve": 0.50,
        "trade_threshold": 0.008,
        "max_per_sector": 2,
        "single_stock_satellite_cap": 0.25,
        "sector_satellite_cap": 0.50,
        "min_adv_musd": 10.0,
        "correlation_threshold": 0.80,
        "research_only_without_approval": True,
    },
    "risk": {
        "base_growth_weight": 0.85,
        "min_growth_weight": 0.05,
        "max_growth_weight": 0.90,
        "high_volatility_percentile": 0.75,
        "extreme_volatility_percentile": 0.90,
        "drawdown_watch": -0.03,
        "drawdown_stress": -0.06,
        "liability_targets": {
            "2026": 0.00, "2027": 0.00, "2028": 0.02, "2029": 0.05,
            "2030": 0.15, "2031": 0.30, "2032": 0.55, "2033": 0.75,
            "2034": 0.80, "2035": 0.85, "2036": 0.90, "2037": 0.92,
            "2038": 0.94, "2039": 0.96, "2040": 0.98, "2041": 1.00, "2042": 1.00
        },
        "time_multipliers": {
            "2026": 1.00, "2027": 1.00, "2028": 0.92, "2029": 0.85,
            "2030": 0.70, "2031": 0.55, "2032": 0.25, "2033": 0.10
        }
    },
    "simulation": {
        "paths": 3000,
        "seed": 42,
        "distribution": "student_t",
        "student_t_df": 6,
        "buckets": ["Growth", "Stability", "FutureLiability"],
        "return_assumptions": {
            "Growth": {"mean": 0.07, "vol": 0.16},
            "Stability": {"mean": 0.03, "vol": 0.05},
            "FutureLiability": {"mean": 0.02, "vol": 0.01}
        },
        "correlation": [
            [1.00, 0.20, 0.10],
            [0.20, 1.00, 0.30],
            [0.10, 0.30, 1.00]
        ],
        "reserve_return": {"mean": 0.02, "vol": 0.01},
        "confidence_thresholds": [0.90, 0.95, 0.975, 0.99],
        "scenarios": ["base", "bull", "bear", "recession", "equity_crash", "rate_shock", "sequence_stress"],
        "commitment_lower_percentile": 10,
        "commitment_upper_percentile": 50,
        "reserve_initial_composition": {"FutureLiability": 0.80, "Stability": 0.20}
    }
}
