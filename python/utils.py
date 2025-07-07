import re
import pandas as pd

def convert_concentration(value):
    """Convert concentration string to numerical value in grams."""
    if pd.isna(value) or value == "":
        return 0.0
    try:
        value = str(value).lower().replace(" ", "")
        match = re.match(r'(\d*\.?\d+)([mnu]?g)?', value)
        if match:
            num = float(match.group(1))
            unit = match.group(2) or ''
            if unit == 'mg':
                return num * 1e-3
            elif unit == 'ug':
                return num * 1e-6
            elif unit == 'ng':
                return num * 1e-9
            return num
        return float(value)
    except ValueError:
        return 0.0