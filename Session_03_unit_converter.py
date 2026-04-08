def convert_kg_to_lb(weight_in_kg):
    return weight_in_kg * 2.2

def convert_km_miles(distance_in_km):
    return distance_in_km*0.6

def celsius_to_fahrenheit(temp_in_celcius):
    return (temp_in_celcius * (9/5)) + 32

def get_valid_number(input_text):
    while True:
        try:
            user_input = input(input_text)
            number = float(user_input)
            return number
        except ValueError:
            print("[Error] Please enter a valid number")

def main():
    print("\n == Smart unit converter ==")
    print("1. Convert kg to lb")
    print("2. Convert km to miles")
    print("3. Convert celsius to Fahrenheit")

    choice = get_valid_number("\nPick a conversion (1-3):")

    if choice == 1:
        weight_in_kg = get_valid_number("Enter the weight in kg:")
        weight_in_lb = convert_kg_to_lb(weight_in_kg)
        print(f"[RESULT] {weight_in_kg} kg = {weight_in_lb} lb\n")
    if choice == 2:
        distance_in_km = get_valid_number("Enter the distance in km:")
        distance_in_miles = convert_km_miles(distance_in_km)
        print(f"[RESULT] {distance_in_km} km = {distance_in_miles} miles\n")
    if choice == 3:
        temp_in_celsius = get_valid_number("Enter the temperature in celsius:")
        temp_in_fahreinheit = celsius_to_fahrenheit(temp_in_celsius)
        print(f"[RESULT] {temp_in_celsius} celsius = {temp_in_fahreinheit} fahrenheit\n")
    else:
        print("[Error] Please pick your choice of 1,2 or 3 \n")

if __name__ == '__main__':
    main()