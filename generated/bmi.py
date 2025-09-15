def bmi(weight_kg: float, height_cm: float) -> float:
    """
    Calculate the Body Mass Index (BMI) given weight in kilograms and height in centimeters.

    Parameters:
    weight_kg (float): Weight in kilograms.
    height_cm (float): Height in centimeters.

    Returns:
    float: The BMI value rounded to 2 decimal places.
    """
    height_m = height_cm / 100
    bmi_value = weight_kg / (height_m**2)
    return round(bmi_value, 2)
