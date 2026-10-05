import math

def get_slope_angle(x, slope_index, data_set_index):

    # ------------------------------------------------------------------
    # Training (10 slopes)
    # ------------------------------------------------------------------
    if data_set_index == 1:
        if slope_index == 1:
            alpha = 4.0 + math.sin(x/100) + math.cos(math.sqrt(2)*x/50)
        elif slope_index == 2:
            alpha = 5.0 - math.sin(x/80) + math.cos(math.sqrt(3)*x/60)
        elif slope_index == 3:
            alpha = 3.5 + 1.5*math.sin(x/70) - math.cos(math.sqrt(2)*x/80)
        elif slope_index == 4:
            alpha = 4.5 + math.sin(x/90) + 0.8*math.cos(math.sqrt(5)*x/50)
        elif slope_index == 5:
            alpha = 3.0 - 1.2*math.sin(x/60) + math.cos(math.sqrt(3)*x/70)
        elif slope_index == 6:
            alpha = 5.5 + math.sin(x/110) - math.cos(math.sqrt(2)*x/40)
        elif slope_index == 7:
            alpha = 4.2 - 0.7*math.sin(x/50) + math.cos(math.sqrt(7)*x/90)
        elif slope_index == 8:
            alpha = 3.8 + math.sin(x/120) + 1.3*math.cos(math.sqrt(5)*x/60)
        elif slope_index == 9:
            alpha = 4.0 - 1.1*math.sin(x/85) - math.cos(math.sqrt(3)*x/55)
        elif slope_index == 10:
            alpha = 3.2 + 2.0*math.sin(x/50) + math.cos(math.sqrt(2)*x/100)

    # ------------------------------------------------------------------
    # Validation (5 slopes) – same statistical family
    # ------------------------------------------------------------------
    elif data_set_index == 2:
        if slope_index == 1:
            alpha = 4.3 - math.sin(x/95) + math.cos(math.sqrt(3)*x/55)
        elif slope_index == 2:
            alpha = 3.7 + math.sin(x/75) - math.cos(math.sqrt(2)*x/65)
        elif slope_index == 3:
            alpha = 5.1 - 0.9*math.sin(x/55) + math.cos(math.sqrt(5)*x/80)
        elif slope_index == 4:
            alpha = 4.0 + 1.4*math.sin(x/95) - math.cos(math.sqrt(7)*x/45)
        elif slope_index == 5:
            alpha = 3.5 + math.sin(x/50) + math.cos(math.sqrt(5)*x/50)

    # ------------------------------------------------------------------
    # Test (5 slopes) – same statistical family
    # ------------------------------------------------------------------
    elif data_set_index == 3:
        if slope_index == 1:
            alpha = 4.1 - math.sin(x/100) + math.cos(math.sqrt(7)*x/50)
        elif slope_index == 2:
            alpha = 3.9 + 1.2*math.sin(x/65) - math.cos(math.sqrt(3)*x/75)
        elif slope_index == 3:
            alpha = 5.0 - math.sin(x/85) + 0.7*math.cos(math.sqrt(2)*x/55)
        elif slope_index == 4:
            alpha = 3.6 + 1.5*math.sin(x/45) + math.cos(math.sqrt(5)*x/90)
        elif slope_index == 5:
            alpha = 4.4 + math.sin(x/70) + math.cos(math.sqrt(7)*x/100)

    return alpha