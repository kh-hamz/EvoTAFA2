"""Post-round truth joins. These values never return to a defense service."""


def summarize(events, risks, threshold, profiles, specialist=False):
    result = {}
    cohorts = {
        "honest": [c for c in risks if not events[c]["compromised"]],
        "active": [c for c in risks if events[c]["scheduled_active"]],
        "inactive_compromised": [c for c in risks if events[c]["compromised"] and not events[c]["scheduled_active"]],
        "honest_concentrated": [c for c in risks if not events[c]["compromised"] and profiles[c]["concentrated"]],
        "honest_declared_specialist_scenario": [c for c in risks if specialist and not events[c]["compromised"]],
    }
    for name, clients in cohorts.items():
        result[name] = {"client_rounds": len(clients), "flags": sum(risks[c] > threshold for c in clients),
                        "flag_rate": sum(risks[c] > threshold for c in clients) / len(clients) if clients else None,
                        "missing_reason": None if clients else "no_cohort_coverage"}
    return result
