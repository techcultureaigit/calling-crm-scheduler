from typing import List, Dict, Any, Optional
from app.core.logger import logger

SURVEY_STEPS = [
    {
        "id": "full_name",
        "field": "full_name",
        "question": "Could you please tell me your full name?",
        "instruction": "Greeting the customer and ask for their full name. If they only give a first name, politely ask if they could share their full name.",
        "skip_if": None
    },
    {
        "id": "age",
        "field": "age",
        "question": "Thank you. And how old are you?",
        "instruction": "Ask for the customer's age. It must be a valid number of years. If they give an age group (like 'in my thirties'), politely ask if they can share their exact age.",
        "skip_if": None
    },
    {
        "id": "employment_status",
        "field": "employment_status",
        "question": "Got it. What is your current employment status? For example, are you employed, self-employed, a student, retired, or unemployed?",
        "instruction": "Determine their employment status. Valid categories: Employed, Self-employed, Student, Retired, Unemployed. If they say something else, map it to the closest category.",
        "skip_if": None
    },
    {
        "id": "annual_income",
        "field": "annual_income",
        "question": "What is your approximate annual income in Rupees?",
        "instruction": "Ask for their annual income in Rupees. Encourage them to give a numeric value or range (e.g., '10 lakhs', 'around 50,000 per month'). Skip this question if their employment status is Unemployed, Student, or Retired (do not ask about income).",
        "skip_if": "employment_status in ['Unemployed', 'Student', 'Retired']"
    },
    {
        "id": "investment_preferences",
        "field": "investment_preferences",
        "question": "Understood. Which investment avenues do you prefer? For example, Mutual Funds, Stocks, Gold, Fixed Deposits, or none of these?",
        "instruction": "Ask for their preferred investment options. They can mention multiple. If they say they do not invest, record 'None' or 'No investments'.",
        "skip_if": None
    },
    {
        "id": "additional_feedback",
        "field": "additional_feedback",
        "question": "Finally, do you have any feedback or suggestions you'd like to share with us today?",
        "instruction": "Ask for any general comments or feedback. Once they answer, thank them and conclude the call.",
        "skip_if": None
    }
]

def get_survey_prompt_context(extracted_data: Dict[str, Any], survey_steps: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generate minimal stateless context for the LLM explaining only the current and next steps.
    """
    context = "SURVEY WORKFLOW STATE:\n"
    current_found = False
    next_pending_found = False
    steps_copy = list(survey_steps) if survey_steps is not None else list(SURVEY_STEPS)
    
    # Virtual Last Step: Farewell
    farewell_step = {
        "id": "farewell",
        "field": "__farewell_acked",
        "question": "[Say Farewell]",
        "instruction": "The survey questions are complete. Thank the user for their time and say the farewell message. Ensure you append [END_CALL] to the end of your response.",
        "skip_if": None
    }
    steps_copy.append(farewell_step)
    
    for step in steps_copy:
        # Check skip condition
        skip = False
        if step.get("skip_if"):
            local_vars = {k: v for k, v in extracted_data.items() if v is not None}
            try:
                skip = eval(step["skip_if"], {"data": local_vars}, local_vars)
                logger.info(f"Evaluated skip_if for step '{step['field']}': {step['skip_if']} => {skip} (locals: {local_vars})")
            except Exception as e:
                logger.error(f"Error evaluating skip_if for step '{step['field']}': {step['skip_if']}. Error: {e}")
                skip = False
        
        if skip:
            continue
            
        field_val = extracted_data.get(step["field"])
        if field_val is not None and field_val != "" and field_val != "Unclear":
            # Step is completed
            continue
        else:
            unclear_msg = ""
            if field_val == "Unclear":
                unclear_msg = "STATUS: UNCLEAR (The user's previous answer was invalid or unclear. You MUST explicitly say you didn't understand, and RE-ASK this question.)\n"

            if not current_found:
                opts = step.get("options", [])
                opt_str = ""
                mcq_rule = ""
                if opts:
                    labels = [o.get("label", o) if isinstance(o, dict) else o for o in opts]
                    opt_str = f"Allowed Options (YOU MUST READ THESE OPTIONS ALOUD TO THE USER): {', '.join(labels)}\n"
                    mcq_rule = "MCQ HANDLING: The user's selected answer MUST be one of the provided options. If the user answers in a different way that resembles an option (e.g. symbol or full name instead of abbreviation), map it to the correct option and explicitly CONFIRM it with the user (e.g. 'So you mean BJP, right?'). If the response cannot be mapped to any option at all, politely re-ask the question and list the options.\n"
                    
                state_instruction = ""
                if not unclear_msg:
                    state_instruction = "STATE TRANSITION INSTRUCTION: If the user just answered a previous question, the backend has successfully recorded it. You may briefly acknowledge it (e.g., 'theek hai', 'dhanyawad'), but DO NOT say you didn't understand. Seamlessly ask the Question above.\n\n"

                context += (
                    f"CURRENT STEP:\n"
                    f"{unclear_msg}"
                    f"Question: '{step['question']}'\n"
                    f"{opt_str}"
                    f"Instructions: {step['instruction']} (DO NOT READ THIS ALOUD)\n"
                    f"{mcq_rule}"
                    f"{state_instruction}"
                )
                current_found = True
            elif not next_pending_found:
                opts = step.get("options", [])
                opt_str = ""
                if opts:
                    labels = [o.get("label", o) if isinstance(o, dict) else o for o in opts]
                    opt_str = f"Allowed Options (YOU MUST READ THESE OPTIONS ALOUD TO THE USER): {', '.join(labels)}\n"
                    
                context += (
                    f"NEXT STEP:\n"
                    f"Question: '{step['question']}'\n"
                    f"{opt_str}"
                    f"Instructions: {step['instruction']} (DO NOT READ THIS ALOUD)\n\n"
                )
                next_pending_found = True
                
    if not current_found:
        context += "STATE: All survey steps completed. Conclude the survey and say goodbye."
    elif not next_pending_found:
        context += "STATE: There are no more steps after the CURRENT STEP. Once the user answers it, the survey is complete."
        
    return context

def get_current_step_field(extracted_data: Dict[str, Any], survey_steps: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """
    Returns the field name of the current active step, so we can save synchronous extracted data to it.
    """
    steps_copy = list(survey_steps) if survey_steps is not None else list(SURVEY_STEPS)
    
    greeting_step = {
        "id": "greeting",
        "field": "__greeting_acked",
        "question": "[Greeting Message Played]",
        "instruction": "The bot just introduced itself. Determine if the user is acknowledging the greeting or giving consent to proceed. If they say 'yes', 'ji', 'hello', 'haan', 'puchiye', or indicate readiness, map to 'ACKNOWLEDGED'.",
        "skip_if": None
    }
    steps_copy.insert(0, greeting_step)
    
    farewell_step = {
        "id": "farewell",
        "field": "__farewell_acked",
        "question": "[Say Farewell]",
        "instruction": "The survey questions are complete. Thank the user for their time and say the farewell message. Ensure you append [END_CALL] to the end of your response.",
        "skip_if": None
    }
    steps_copy.append(farewell_step)
    
    for step in steps_copy:
        skip = False
        if step.get("skip_if"):
            local_vars = {k: v for k, v in extracted_data.items() if v is not None}
            try:
                skip = eval(step["skip_if"], {"data": local_vars}, local_vars)
            except Exception:
                skip = False
                
        if skip:
            continue
            
        field_val = extracted_data.get(step["field"])
        if field_val is not None and field_val != "" and field_val != "Unclear":
            continue
        else:
            return step["field"]
    
    return None

def get_current_step_dict(extracted_data: Dict[str, Any], survey_steps: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Returns the dictionary of the current active step.
    """
    field = get_current_step_field(extracted_data, survey_steps)
    if not field:
        return None
        
    steps_copy = list(survey_steps) if survey_steps is not None else list(SURVEY_STEPS)
    
    greeting_step = {
        "id": "greeting",
        "field": "__greeting_acked",
        "question": "[Greeting Message Played]",
        "instruction": "The bot just introduced itself. Determine if the user is acknowledging the greeting or giving consent to proceed. If they say 'yes', 'ji', 'hello', 'haan', 'puchiye', or indicate readiness, map to 'ACKNOWLEDGED'.",
        "skip_if": None
    }
    steps_copy.insert(0, greeting_step)
    
    farewell_step = {
        "id": "farewell",
        "field": "__farewell_acked",
        "question": "[Say Farewell]",
        "instruction": "Say the farewell message.",
        "skip_if": None
    }
    steps_copy.append(farewell_step)
    
    for step in steps_copy:
        if step["field"] == field:
            return step
    return None

def is_survey_complete(extracted_data: Dict[str, Any], survey_steps: Optional[List[Dict[str, Any]]] = None) -> bool:
    """
    Check if all non-skipped steps have data.
    """
    steps = list(survey_steps) if survey_steps is not None else list(SURVEY_STEPS)
    
    # Virtual Last Step: Farewell
    farewell_step = {
        "id": "farewell",
        "field": "__farewell_acked",
        "question": "[Say Farewell]",
        "instruction": "The survey questions are complete. Thank the user for their time and say the farewell message.",
        "skip_if": None
    }
    steps.append(farewell_step)
    
    for step in steps:
        skip = False
        if step.get("skip_if"):
            local_vars = {k: v for k, v in extracted_data.items() if v is not None}
            try:
                skip = eval(step["skip_if"], {"data": local_vars}, local_vars)
            except Exception:
                skip = False
        if skip:
            continue
        val = extracted_data.get(step["field"])
        if val is None or val == "" or val == "Unclear":
            return False
    return True

def clean_internal_fields(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a copy of the extracted_data with all internal fields (starting with __) removed.
    """
    if not extracted_data:
        return {}
    return {k: v for k, v in extracted_data.items() if not k.startswith("__")}

