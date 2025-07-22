# Plant Care Assistant

A rule-based AI system for diagnosing plant health issues.

## How to Run
```
python plant_care_assistant.py
```

## Requirements
- Python 3.6+
- No external dependencies

## Author
Jeremiah Williams - CAP320 Assignment


# Building a Rule-Based AI System with Python - Plant Care Assistant

## Part 1: Initial Project Ideas

### 1. Project Idea 1: Personal Finance Advisor
* **Description:** A system that provides budgeting advice and financial recommendations based on user's income, expenses, and financial goals.
* **Rule-Based Approach:**
  * The system evaluates financial ratios and provides targeted advice.
  * IF income > expenses AND savings_rate < 20% THEN recommend increasing savings
  * IF debt_to_income > 30% THEN prioritize debt reduction
  * IF emergency_fund < 3_months_expenses THEN recommend building emergency fund first

### 2. Project Idea 2: Plant Care Assistant
* **Description:** A diagnostic tool that helps users troubleshoot problems with their houseplants and provides care recommendations.
* **Rule-Based Approach:**
  * The system analyzes plant symptoms and environmental conditions to diagnose issues.
  * IF leaves_yellowing AND soil_wet THEN "overwatering - reduce watering frequency"
  * IF leaves_drooping AND soil_dry THEN "underwatering - water immediately"
  * IF brown_leaf_tips AND low_humidity THEN "increase humidity around plant"

### 3. Project Idea 3: Study Schedule Optimizer
* **Description:** A system that creates personalized study schedules based on available time, subjects, difficulty levels, and upcoming exams.
* **Rule-Based Approach:**
  * The system allocates study time using priority-based rules.
  * IF exam_date < 7_days THEN allocate 40% of study time to that subject
  * IF subject_difficulty = "high" THEN recommend shorter study sessions (25-30 min)
  * IF available_time < 2_hours THEN focus on review rather than new material

**Chosen Idea:** Plant Care Assistant

**Justification:** I chose the Plant Care Assistant because it addresses a practical, everyday problem that many people face. The rule-based logic is clear and intuitive, making it easy to understand how traditional AI systems worked before machine learning. Additionally, plant care has well-established guidelines that translate naturally into if-then rules, making it perfect for demonstrating rule-based AI concepts.

## Part 2: Rules/Logic for the Chosen System

The **Plant Care Assistant** system will follow these rules organized by priority:

### 1. **Critical Watering Issues (Highest Priority):**
   * **IF** leaves_yellowing = True AND soil_condition = "wet" → **Diagnose overwatering** and recommend immediate drainage
   * **IF** leaves_drooping = True AND soil_condition = "wet" → **Warn of possible root rot** and suggest root inspection
   * **IF** leaves_drooping = True AND soil_condition = "dry" → **Diagnose underwatering** and recommend immediate watering

### 2. **Light-Related Problems:**
   * **IF** leaves_pale = True AND light_level = "low" → **Diagnose insufficient light** and suggest brighter location
   * **IF** leaves_scorched = True AND light_level = "high" → **Diagnose too much direct sunlight** and recommend indirect light
   * **IF** plant_stretching = True → **Diagnose light-seeking behavior** and suggest rotation/increased exposure

### 3. **Environmental Issues:**
   * **IF** brown_leaf_tips = True → **Diagnose humidity/water quality issues** and recommend filtered water
   * **IF** leaves_dropping = True AND recent_move = True → **Diagnose transplant shock** and recommend patience
   * **IF** pest_signs = True → **Diagnose pest infestation** and recommend isolation/treatment

### 4. **Plant-Specific Care Rules:**
   * **IF** plant_type = "succulent" AND soil_condition = "wet" → **Provide succulent-specific overwatering warning**
   * **IF** plant_type = "fern" AND (brown_leaf_tips OR leaves_drooping) → **Recommend humidity solutions for ferns**
   * **IF** plant_type = "snake_plant" AND leaves_yellowing → **Remind about drought tolerance**

### 5. **General Maintenance:**
   * **IF** roots_visible = True → **Recommend repotting**
   * **IF** no issues detected → **Confirm plant appears healthy**

## Part 3: Test Results

Sample input and output from the system:

### Test 1: Overwatered Succulent
```
Plant type: Succulent
Are the leaves turning yellow? y
What's the soil condition? 3 (Wet/soggy)

🔍 IDENTIFIED ISSUES:
1. 🚨 OVERWATERING DETECTED
2. 🌵 SUCCULENT OVERWATERING

💡 RECOMMENDATIONS:
1. Stop watering immediately and ensure proper drainage. Allow soil to dry out completely.
2. Succulents need excellent drainage. Water only when soil is completely dry.
```

### Test 2: Light-Starved Plant
```
Plant type: Pothos
Are the leaves pale or losing color? y
Is the plant growing tall and leggy? y
What's the light level? 1 (Low light)

🔍 IDENTIFIED ISSUES:
1. ☀️ INSUFFICIENT LIGHT
2. 🌱 SEEKING MORE LIGHT

💡 RECOMMENDATIONS:
1. Move to a brighter location with indirect sunlight or add a grow light.
2. Rotate plant regularly and gradually increase light exposure.
```

### Test 3: Healthy Plant
```
Plant type: Snake Plant
All symptoms: n
Soil condition: Dry
Light level: Medium

🔍 IDENTIFIED ISSUES:
1. ✅ PLANT APPEARS HEALTHY

💡 RECOMMENDATIONS:
1. Continue current care routine. Monitor for any changes.
```

## Part 4: Reflection

### Project Overview:
This project involved designing a practical, rule-based system to diagnose plant health issues and provide care recommendations. The system uses logical conditions and prioritized rule sets to evaluate user-provided symptoms against established plant care principles, demonstrating how AI systems functioned before machine learning became widespread.

### How the System Works:
My Plant Care Assistant operates using a hierarchical rule-based approach that processes user inputs through six organized rule sets. The system first collects information about plant type and symptoms using multiple `input()` functions, then applies `if-elif-else` statements to diagnose problems in order of severity.

The system prioritizes life-threatening issues like overwatering and root rot first, since these can quickly kill a plant. It then evaluates environmental factors (light, humidity), pest problems, and provides plant-specific recommendations. Each rule follows clear IF-THEN logic: IF certain conditions are met (like yellowing leaves AND wet soil), THEN provide specific diagnosis and recommendations (overwatering detected, stop watering immediately).

What makes this system effective is its ability to provide different advice for different plant types. The same symptom might indicate different problems - wet soil is dangerous for succulents but normal for ferns. This demonstrates how rule-based systems can replicate expert knowledge through logical decision trees.

### Challenges Encountered:
* **Rule Prioritization:** Determining the correct order for evaluating rules was challenging. I had to ensure critical issues like root rot were addressed before minor problems like fertilizing needs.
* **Symptom Overlap:** Many plant problems share similar symptoms. Designing rules that could differentiate between causes (like distinguishing overwatering from underwatering when both cause drooping) required careful condition checking.
* **Input Validation:** Ensuring the AI helped create robust input handling while keeping the system simple and rule-based rather than overly complex was an ongoing challenge.
* **Comprehensive Coverage:** Balancing thoroughness with simplicity - wanting enough rules for accurate diagnosis without making the system unwieldy. The AI helped refine this balance by suggesting logical groupings and priority systems.

The iterative process with the AI assistant was crucial for developing a system that truly demonstrates rule-based AI principles while remaining practical and user-friendly.
