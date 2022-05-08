RangeOfVar = [0.1, 10]

old_value = 0

# coeff +  variance decrease
# coeff -  variance increase
def adjust_var_coeff(variance, coeff):
    if variance > 100:
      if coeff > 10:
        return 10
      if coeff < 0:
        return 1
      return coeff*2
    if variance > 10:
      if coeff > 10:
        return 5
      if coeff < 0:
        return 1
      return coeff
    if variance > 1:
        if coeff > 10:
            return 5
        if coeff < 0:
            return -0.1
        return coeff
    if variance > 0.1:
        if coeff > 0.1:
            return coeff/2
        if coeff < -10:
            return -10
        if -10 <= coeff < -1:
            return coeff/2
        return -1
    if variance < 0.01:
        if coeff > 0:
            return -5
        if -5 < coeff < 0:
            return coeff * 2
        if -10 < coeff <= -5:
            return coeff - 1
        if coeff <= -10:
            return -10 
        if coeff == 0:
            return -5
        return coeff
    return 0
    
                
