ADMIN_PAGES = (
    "Home",
    "Overview",
    "Hospitals",
    "Models",
    "Privacy",
    "Prediction",
    "Retraining",
    "Federated Operations",
    "Experimental Evaluation",
)

HOSPITAL_PAGES = (
    "Home",
    "Local Dataset",
    "Local Training",
    "Federated Participation",
    "Prediction",
    "Prediction History",
)


def pages_for_role(role):
    if role == "admin":
        return ADMIN_PAGES
    if role == "hospital":
        return HOSPITAL_PAGES
    raise ValueError("Unsupported account role.")


def page_is_allowed(role, page):
    try:
        return page in pages_for_role(role)
    except ValueError:
        return False
