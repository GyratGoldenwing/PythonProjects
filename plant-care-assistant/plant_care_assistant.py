#!/usr/bin/env python3
"""
Plant Care Assistant - Rule-Based AI System
A diagnostic tool that helps users troubleshoot houseplant problems
and provides personalized care recommendations.
"""

def get_plant_info():
    """
    Collect basic information about the user's plant using input() function.
    This implements the user input requirement for the rule-based system.
    """
    print("=== Plant Care Assistant ===")
    print("Let's diagnose your plant's health!\n")
    
    # Get plant type using input() function and input validation
    plant_types = ["succulent", "fern", "snake_plant", "pothos", "spider_plant", "other"]
    print("What type of plant do you have?")
    for i, plant_type in enumerate(plant_types, 1):
        print(f"{i}. {plant_type.title()}")
    
    # Input validation loop using if-elif-else logic
    while True:
        try:
            choice = int(input("\nEnter number (1-6): "))  # USER INPUT using input() function
            if 1 <= choice <= len(plant_types):
                plant_type = plant_types[choice - 1]
                break
            else:
                print("Please enter a number between 1 and 6.")
        except ValueError:
            print("Please enter a valid number.")
    
    return plant_type

def assess_symptoms():
    """
    Collect information about plant symptoms using input() function.
    This function gathers all the data needed for rule-based decision making.
    """
    symptoms = {}
    
    # Leaf condition questions - using input() function to collect user data
    print("\nPlease answer the following questions about your plant's symptoms:")
    
    # Each input() call collects boolean data for rule conditions
    symptoms['leaves_yellowing'] = input("Are the leaves turning yellow? (y/n): ").lower().startswith('y')
    symptoms['leaves_drooping'] = input("Are the leaves drooping or wilting? (y/n): ").lower().startswith('y')
    symptoms['leaves_pale'] = input("Are the leaves pale or losing color? (y/n): ").lower().startswith('y')
    symptoms['leaves_scorched'] = input("Are there brown/burnt spots on leaves? (y/n): ").lower().startswith('y')
    symptoms['brown_leaf_tips'] = input("Do the leaf tips have brown edges? (y/n): ").lower().startswith('y')
    symptoms['leaves_dropping'] = input("Is the plant dropping leaves? (y/n): ").lower().startswith('y')
    
    # Growth and structure questions
    symptoms['plant_stretching'] = input("Is the plant growing tall and leggy? (y/n): ").lower().startswith('y')
    symptoms['roots_visible'] = input("Are roots growing out of drainage holes? (y/n): ").lower().startswith('y')
    
    # Environmental conditions - using if-elif-else for input validation
    print("\nWhat's the soil condition?")
    print("1. Dry/crumbly")
    print("2. Moist but not wet")
    print("3. Wet/soggy")
    
    # Input validation using if-elif-else statements
    while True:
        try:
            soil_choice = int(input("Enter number (1-3): "))  # USER INPUT
            if soil_choice == 1:
                symptoms['soil_condition'] = "dry"
                break
            elif soil_choice == 2:
                symptoms['soil_condition'] = "moist"
                break
            elif soil_choice == 3:
                symptoms['soil_condition'] = "wet"
                break
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Light level assessment using if-elif-else logic
    print("\nWhat's the light level where your plant is located?")
    print("1. Low light (far from windows)")
    print("2. Medium light (near window, indirect)")
    print("3. High light (direct sunlight)")
    
    while True:
        try:
            light_choice = int(input("Enter number (1-3): "))  # USER INPUT
            if light_choice == 1:
                symptoms['light_level'] = "low"
                break
            elif light_choice == 2:
                symptoms['light_level'] = "medium"
                break
            elif light_choice == 3:
                symptoms['light_level'] = "high"
                break
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Additional factors - more input() function usage
    symptoms['recent_move'] = input("Have you recently moved or repotted the plant? (y/n): ").lower().startswith('y')
    symptoms['pest_signs'] = input("Do you see any bugs or sticky residue? (y/n): ").lower().startswith('y')
    
    return symptoms

def diagnose_plant(plant_type, symptoms):
    """
    Apply rule-based logic to diagnose plant problems using if-elif-else statements.
    This function implements the core AI logic of the system.
    """
    diagnoses = []
    recommendations = []
    
    # RULE SET 1: Critical watering issues (highest priority)
    # These rules check for life-threatening conditions first
    
    # Rule 1A: IF yellowing leaves AND wet soil THEN overwatering
    if symptoms['leaves_yellowing'] and symptoms['soil_condition'] == "wet":
        diagnoses.append("🚨 OVERWATERING DETECTED")
        recommendations.append("Stop watering immediately and ensure proper drainage. Allow soil to dry out completely.")
    
    # Rule 1B: IF drooping leaves AND wet soil THEN possible root rot
    if symptoms['leaves_drooping'] and symptoms['soil_condition'] == "wet":
        diagnoses.append("🚨 POSSIBLE ROOT ROT")
        recommendations.append("Check roots for black/mushy areas. If found, trim affected roots and repot in fresh soil.")
    
    # Rule 1C: IF drooping leaves AND dry soil THEN underwatering
    if symptoms['leaves_drooping'] and symptoms['soil_condition'] == "dry":
        diagnoses.append("💧 UNDERWATERING")
        recommendations.append("Water thoroughly until water drains from the bottom. Check soil moisture regularly.")
    
    # RULE SET 2: Light-related problems
    # These rules diagnose lighting issues based on symptoms and current light levels
    
    # Rule 2A: IF pale leaves AND low light THEN insufficient light
    if symptoms['leaves_pale'] and symptoms['light_level'] == "low":
        diagnoses.append("☀️ INSUFFICIENT LIGHT")
        recommendations.append("Move to a brighter location with indirect sunlight or add a grow light.")
    
    # Rule 2B: IF scorched leaves AND high light THEN too much direct sun
    if symptoms['leaves_scorched'] and symptoms['light_level'] == "high":
        diagnoses.append("🔥 TOO MUCH DIRECT SUNLIGHT")
        recommendations.append("Move to a location with bright, indirect light to prevent leaf burn.")
    
    # Rule 2C: IF plant stretching THEN seeking more light
    if symptoms['plant_stretching']:
        diagnoses.append("🌱 SEEKING MORE LIGHT")
        recommendations.append("Rotate plant regularly and gradually increase light exposure.")
    
    # RULE SET 3: Environmental issues
    # These rules check for humidity, air quality, and environmental stress
    
    # Rule 3A: IF brown leaf tips THEN humidity or water quality issues
    if symptoms['brown_leaf_tips']:
        diagnoses.append("💨 LOW HUMIDITY OR WATER QUALITY ISSUES")
        recommendations.append("Increase humidity with a pebble tray and use filtered or distilled water.")
    
    # Rule 3B: IF leaves dropping AND recent move THEN transplant shock
    if symptoms['leaves_dropping'] and symptoms['recent_move']:
        diagnoses.append("📦 TRANSPLANT SHOCK")
        recommendations.append("This is normal after moving/repotting. Maintain consistent care and be patient.")
    
    # Rule 3C: IF pest signs THEN infestation
    if symptoms['pest_signs']:
        diagnoses.append("🐛 PEST INFESTATION")
        recommendations.append("Isolate plant immediately. Clean with insecticidal soap or neem oil.")
    
    # RULE SET 4: Plant-specific care recommendations
    # These rules provide specialized advice based on plant type using if-elif-else logic
    
    # Rule 4A: Succulent-specific rules
    if plant_type == "succulent":
        if symptoms['soil_condition'] == "wet":
            diagnoses.append("🌵 SUCCULENT OVERWATERING")
            recommendations.append("Succulents need excellent drainage. Water only when soil is completely dry.")
    
    # Rule 4B: Fern-specific rules
    elif plant_type == "fern":
        if symptoms['brown_leaf_tips'] or symptoms['leaves_drooping']:
            diagnoses.append("🌿 FERN HUMIDITY NEEDS")
            recommendations.append("Ferns love humidity. Use a humidifier or group with other plants.")
    
    # Rule 4C: Snake plant-specific rules
    elif plant_type == "snake_plant":
        if symptoms['leaves_yellowing']:
            diagnoses.append("🐍 SNAKE PLANT CARE")
            recommendations.append("Snake plants are drought-tolerant. Water only when soil is dry.")
    
    # RULE SET 5: General maintenance rules
    # These rules check for common maintenance needs
    
    # Rule 5A: IF roots visible THEN needs repotting
    if symptoms['roots_visible']:
        diagnoses.append("🪴 NEEDS REPOTTING")
        recommendations.append("Plant is root-bound. Repot into a container 1-2 inches larger.")
    
    # RULE SET 6: Default case - if no issues detected
    # This implements the ELSE condition of the rule-based system
    if not diagnoses:
        diagnoses.append("✅ PLANT APPEARS HEALTHY")
        recommendations.append("Continue current care routine. Monitor for any changes.")
    
    return diagnoses, recommendations

def provide_plant_specific_care(plant_type):
    """Provide general care tips based on plant type"""
    care_tips = {
        "succulent": [
            "Water only when soil is completely dry (every 1-2 weeks)",
            "Provide bright, indirect sunlight",
            "Use well-draining cactus soil",
            "Avoid overwatering - it's the #1 killer of succulents"
        ],
        "fern": [
            "Keep soil consistently moist but not waterlogged",
            "Provide high humidity (50%+ preferred)",
            "Place in bright, indirect light",
            "Mist regularly or use a humidity tray"
        ],
        "snake_plant": [
            "Water every 2-3 weeks, allowing soil to dry between waterings",
            "Tolerates low light but prefers bright, indirect light",
            "Very drought-tolerant - better to underwater than overwater",
            "Clean leaves occasionally to remove dust"
        ],
        "pothos": [
            "Water when top inch of soil is dry",
            "Thrives in medium to bright, indirect light",
            "Pinch back vines to encourage bushier growth",
            "Can be propagated easily in water"
        ],
        "spider_plant": [
            "Water when soil surface is dry",
            "Prefers bright, indirect light",
            "Remove brown tips with clean scissors",
            "Produces baby plants that can be propagated"
        ],
        "other": [
            "Research your specific plant's needs",
            "Most houseplants prefer indirect light",
            "Check soil moisture before watering",
            "Ensure proper drainage in all pots"
        ]
    }
    
    return care_tips.get(plant_type, care_tips["other"])

def main():
    """
    Main program function that orchestrates the rule-based AI system.
    This demonstrates the complete flow: input → rule processing → output
    """
    try:
        # STEP 1: USER INPUT - Collect plant information using input() function
        plant_type = get_plant_info()
        print(f"\nGreat! You have a {plant_type}.")
        
        # STEP 2: USER INPUT - Assess symptoms using multiple input() calls
        print("\nNow let's assess your plant's current condition:")
        symptoms = assess_symptoms()
        
        # STEP 3: RULE-BASED PROCESSING - Apply if-elif-else logic to diagnose
        print("\n" + "="*50)
        print("PLANT DIAGNOSIS RESULTS")
        print("="*50)
        
        # Call the rule-based diagnosis function
        diagnoses, recommendations = diagnose_plant(plant_type, symptoms)
        
        # STEP 4: OUTPUT - Display text responses based on rule results
        print("\n🔍 IDENTIFIED ISSUES:")
        for i, diagnosis in enumerate(diagnoses, 1):
            print(f"{i}. {diagnosis}")  # TEXT OUTPUT based on rules
        
        print("\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(recommendations, 1):
            print(f"{i}. {recommendation}")  # TEXT OUTPUT based on rules
        
        # STEP 5: OUTPUT - Provide additional care tips
        print(f"\n🌱 GENERAL CARE TIPS FOR {plant_type.upper()}:")
        care_tips = provide_plant_specific_care(plant_type)
        for i, tip in enumerate(care_tips, 1):
            print(f"{i}. {tip}")  # TEXT OUTPUT based on plant type
        
        # Final output messages
        print("\n" + "="*50)
        print("Thank you for using the Plant Care Assistant!")
        print("Remember to monitor your plant and reassess if symptoms persist.")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please try again.")

if __name__ == "__main__":
    main()