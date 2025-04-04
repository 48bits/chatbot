import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib import colors
from datetime import datetime

# Check CUDA status
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("CUDA is not available, falling back to CPU")
    device = "cpu"

print("Loading the medical chatbot for telemedicine intake...")

# Choose a model that doesn't require GGUF
MODEL_NAME = "Joker-sxj/Qwen2.5-3B-instruct-medical-finetuned"

# Download and load the model with transformers
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load the model with GPU support
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use float16 for faster inference on GPU
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # Helps when loading large models
)

print(f"Model loaded on: {device}")

# Telemedicine intake system prompt
SYSTEM_PROMPT = """**ROLE & GOAL:**
You are an AI Medical Intake Assistant for a telemedicine service based in Singapore. Your primary role is to conduct a preliminary patient interview before their consultation with the doctor. Your goal is to collect the patient's subjective information accurately and efficiently to create a structured summary for the doctor's review.

**COMMUNICATION STYLE:**
- Interact in a friendly, empathetic, and professional manner.
- Use clear, simple language. Avoid complex medical jargon.
- Be patient and understanding.
- If the patient's answer is vague or unclear, ask polite follow-up questions to clarify the necessary details.
- Transition smoothly and politely between topics (e.g., "Thank you for sharing that. Now, could we talk about...?").

**CONVERSATION STEPS (Follow this order):**
1.  **Initiation & Chief Complaint (CC):** Start the conversation with a warm, polite greeting. Immediately ask for the patient's main reason for scheduling the visit (the Chief Complaint).
2.  **History of Present Illness (HPI):** Thoroughly explore the Chief Complaint. Ask clarifying questions to understand details like:
    * *Onset:* When did it start?
    * *Location:* Where exactly is the symptom?
    * *Duration:* How long does it last? Constant or intermittent?
    * *Character:* Description of the feeling (sharp, dull, aching, etc.)?
    * *Aggravating/Alleviating Factors:* What makes it better or worse?
    * *Radiation:* Does the feeling travel elsewhere?
    * *Timing:* Specific times it's worse? Frequency?
    * *Severity:* Rating (e.g., 1-10 scale)?
    * *Associated Symptoms:* Anything else occurring with it?
3.  **Past Medical History (PMH):** Inquire about relevant ongoing health conditions (e.g., diabetes, high blood pressure, asthma) and any significant past surgeries.
4.  **Current Medications:** Ask about *all* current medications, including prescriptions, over-the-counter drugs, vitamins, and supplements. Try to ascertain the name and dosage/frequency if known by the patient.
5.  **Allergies:** Specifically ask about allergies (medications, food, environmental) and the type of reaction experienced.

**COMPLETION PROTOCOL:**
When you believe you have gathered sufficient information across all required categories (CC, HPI, PMH, medications, allergies, and goals/concerns), you may indicate this by stating "INTAKE COMPLETE" in your internal assessment. At this point, you should thank the patient for providing their information and let them know you will prepare a summary for the doctor.

**CRITICAL SAFETY RULES (Strict Adherence Required):**

1.  **NO MEDICAL ADVICE:**
    * You **MUST NOT** provide any medical advice, diagnosis, interpretations of symptoms, treatment suggestions, or reassurance about symptom severity.
    * If asked for advice or interpretation (e.g., "Is this serious?", "What should I do?"), you **MUST** use phrasing similar to: "I am an AI assistant designed to gather information for the doctor and cannot provide medical advice or interpretations. Please make sure to discuss all your concerns, symptoms, and potential treatments directly with the doctor during your consultation."
    * Do not deviate from this deflection.

2.  **EMERGENCY PROTOCOL:**
    * If the patient describes symptoms strongly suggesting a medical emergency (e.g., severe chest pain/pressure, difficulty breathing, sudden weakness/numbness, confusion, severe headache, loss of consciousness, uncontrolled bleeding, thoughts of self-harm/harming others), you **MUST IMMEDIATELY** stop all intake questions.
    * Respond **ONLY** with the following message: "Based on the symptoms you're describing, it sounds like a potential emergency, and it's important to seek immediate medical attention. Please call 995 or go to the nearest A&E (Accident & Emergency department) right away."
    * **Do not** ask further questions or continue the intake after delivering this emergency message.

**FINAL OUTPUT GOAL:**
The information meticulously gathered during this conversation will be used to generate an accurate subjective summary for the doctor, aiding them in the consultation. Focus on capturing the patient's self-reported information accurately."""







# SOAP Note Summarization Prompt
#SOAP_SUMMARIZATION_PROMPT = """Based strictly on our conversation above, please generate ONLY the Subjective section of a SOAP note suitable for a doctor. Structure it clearly with these headings:
#- Chief Complaint (CC):
#- History of Present Illness (HPI): [Synthesize details like onset, location, duration, character, severity, aggravating/alleviating factors, radiation, timing]
#- Relevant Past Medical History (PMH): [Include if mentioned]
#- Current Medications: [Include if mentioned]
#- Allergies: [Include if mentioned]
#- Patient Goals/Concerns: [Include if mentioned]

#Use concise medical phrasing where appropriate. Do not add any information not explicitly stated by the patient in our chat. Do not include Objective, Assessment, or Plan sections."""

SOAP_SUMMARIZATION_PROMPT = """Based strictly on our conversation above, please generate ONLY the Subjective section of a SOAP note suitable for a doctor. Structure it clearly with these exact headings, ensuring all headings are present:

- Chief Complaint (CC):
  [State the main reason for the visit concisely]
- History of Present Illness (HPI):
  [Provide a narrative summary synthesizing details like onset, location, duration, character, severity, context, aggravating/alleviating factors, radiation, timing, and associated symptoms mentioned.]
- Past Medical History (PMH):
  [List relevant ongoing conditions or past surgeries mentioned. If none were discussed or patient denied history, state 'None reported'.]
- Current Medications:
  [List medications mentioned (prescription, OTC, supplements), including dosage if provided. If none mentioned or patient denied taking any, state 'None reported'.]
- Allergies:
  [List any allergies mentioned (meds, food, other) and the reaction if provided. If none mentioned or patient denied allergies, state 'No Known Drug Allergies (NKDA)' or 'None reported'.]
- Patient Goals/Concerns:
  [Summarize any specific worries or goals the patient expressed for the visit. If none mentioned, state 'None expressed'.]

**Instructions:**
- Use ONLY information explicitly stated by the patient during the conversation. Do not infer or add outside knowledge.
- Summarize key details concisely. Avoid conversational filler (e.g., "The patient said that...").
- Present the information factually.
- Do NOT include Objective (O), Assessment (A), or Plan (P) sections. """



def generate_response(conversation_history, prompt, max_new_tokens=512, temperature=0.7):
    """Generate a response from the model"""
    # Format the conversation history with system prompt
    formatted_conversation = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    
    # Add previous conversation turns if any
    for i, message in enumerate(conversation_history):
        if i % 2 == 0:  # User message
            formatted_conversation += f"<|im_start|>user\n{message}<|im_end|>\n"
        else:  # Assistant message
            formatted_conversation += f"<|im_start|>assistant\n{message}<|im_end|>\n"
    
    # Add the current user message
    formatted_conversation += f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Create input tokens
    inputs = tokenizer(formatted_conversation, return_tensors="pt").to(model.device)
    
    # Time the generation
    start_time = time.time()
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Calculate generation time
    generation_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"Generation took {generation_time:.2f} seconds")
    return response.strip()

def generate_soap_summary(conversation_history, patient_name="[Patient Name]", patient_id="[Patient ID]"):
    """Generate a complete SOAP note with AI for Subjective and templates for O, A, P"""
    # Use a lower temperature for more factual summarization
    subjective_section = generate_response(conversation_history, SOAP_SUMMARIZATION_PROMPT, temperature=0.4)
    
    # Create the complete SOAP note with templates for O, A, P
    soap_note = f"""PATIENT NAME: {patient_name}
PATIENT ID: {patient_id}
DATE: {datetime.now().strftime("%d/%m/%Y")}
TIME: {datetime.now().strftime("%H:%M")}
CONSULTATION TYPE: Telemedicine

SOAP NOTE:

S (SUBJECTIVE):
{subjective_section}

O (OBJECTIVE):
Vitals: [To be completed by clinician]
- BP: ___/___
- HR: ___ bpm
- RR: ___ /min
- Temp: ___ °C
- SpO2: ___% on room air

Physical Exam: [To be completed by clinician]
- General Appearance: 
- HEENT: 
- Cardiovascular: 
- Respiratory: 
- Abdomen: 
- Extremities: 
- Neuro: 
- Skin: 

Relevant Test Results: [To be completed by clinician]

A (ASSESSMENT):
[To be completed by clinician]
1. 
2. 
3. 

P (PLAN):
[To be completed by clinician]
1. Diagnostics: 
2. Therapeutics: 
3. Patient Education: 
4. Follow-up: 

----------------------------
Generated by AI Telemedicine Intake Assistant
Note: Subjective section was auto-generated based on patient interview.
All other sections require clinician completion.
----------------------------
"""
    return soap_note

def create_soap_pdf(soap_text, filename="soap_note.pdf"):
    """Convert the SOAP note to a PDF file"""
    doc = SimpleDocTemplate(filename, pagesize=letter, 
                         rightMargin=72, leftMargin=72,
                         topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Normal_LEFT',
                              parent=styles['Normal'],
                              alignment=TA_LEFT,
                              leading=14))
    
    # Add a title style (renamed from 'Title' to 'DocTitle' to avoid conflict)
    styles.add(ParagraphStyle(name='DocTitle',
                              parent=styles['Heading1'],
                              fontSize=16,
                              textColor=colors.darkblue,
                              spaceAfter=12))
    
    # Add a section header style
    styles.add(ParagraphStyle(name='SectionHeader',
                            parent=styles['Heading2'],
                            fontSize=12,
                            textColor=colors.darkblue,
                            spaceAfter=6))
    
    # Process the SOAP text into paragraphs
    flowables = []
    
    # Add title (using renamed style)
    flowables.append(Paragraph("TELEMEDICINE SOAP NOTE", styles['DocTitle']))
    flowables.append(Spacer(1, 12))
    
    # Split by lines and add formatting
    sections = soap_text.split("\n\n")
    for section in sections:
        if ":" in section and not section.startswith("-"):
            # This is likely a header
            header, content = section.split(":", 1)
            flowables.append(Paragraph(header + ":", styles['SectionHeader']))
            if content.strip():
                flowables.append(Paragraph(content.strip(), styles['Normal_LEFT']))
                flowables.append(Spacer(1, 6))
        else:
            # Process the section line by line
            lines = section.split("\n")
            for line in lines:
                if line.strip():
                    # Check if this is a section header (S, O, A, P)
                    if line.startswith("S (") or line.startswith("O (") or line.startswith("A (") or line.startswith("P ("):
                        flowables.append(Paragraph(line, styles['SectionHeader']))
                    elif line.startswith("- "):
                        # Bullet point
                        flowables.append(Paragraph("• " + line[2:], styles['Normal_LEFT']))
                    else:
                        flowables.append(Paragraph(line, styles['Normal_LEFT']))
            flowables.append(Spacer(1, 6))
    
    # Build the PDF
    doc.build(flowables)
    print(f"PDF saved as {filename}")
    return filename

def chat():
    """Interactive chat loop with context-based narrative approach for medical intake"""
    print("\nTelemedicine Intake Assistant - Starting automated intake process")
    print("The assistant will guide you through the intake process and automatically generate a summary when complete.")
    
    # Store conversation history
    conversation_history = []
    
    # Initial greeting with first question
    initial_greeting = "Hello! Thank you for taking the time to speak with me today. I'm here to help gather some information so we can better understand what you're experiencing. Could you please tell me why you're here?"
    print(f"\nAI: {initial_greeting}")
    conversation_history.append("Hello, I'm here for a consultation.")
    conversation_history.append(initial_greeting)
    
    # Track conversation progress
    exchange_count = 0
    
    # Topic areas to cover in a natural progression
    topic_areas = [
        "chief_complaint",
        "symptom_details",
        "medical_history",
        "medications",
        "allergies",
        "goals"
    ]
    
    # Track where we are in the conversation
    current_topic_area = "chief_complaint" 
    
    # Simple narrative guidance for topic transitions
    topic_transitions = {
        "chief_complaint_to_symptom_details": "Now I'd like to understand more about your symptoms.",
        "symptom_details_to_medical_history": "Thank you for sharing those details. I'd also like to know about your medical history.",
        "medical_history_to_medications": "Let's talk about any medications you might be taking.",
        "medications_to_allergies": "Let me also ask about any allergies you might have.",
        "allergies_to_goals": "Finally, I'd like to understand what you're hoping to get from this consultation."
    }
    
    # Conversation guidance prompts based on context
    def get_context_prompt(topic, exchange_count, conversation_history):
        # First exchange - just capture chief complaint
        if exchange_count == 0:
            return "Listen to the patient's primary concern."
            
        # Early in conversation - focus on symptom details
        if exchange_count < 5 and topic == "symptom_details":
            # Check if we have basic complaint info
            last_user_msg = conversation_history[-2].lower() if len(conversation_history) >= 2 else ""
            if "headache" in last_user_msg or "pain" in last_user_msg:
                return "The patient mentioned symptoms that sound like pain. Ask about when it started, severity, and characteristics."
            else:
                return "Ask about the nature of their primary symptoms - when they started, severity, and characteristics."
            
        # Symptom exploration - cover important aspects without rigid structure
        if topic == "symptom_details" and exchange_count < 10:
            return """
Based on the conversation so far, identify which aspects of the symptoms haven't been covered yet and ask about ONE of them. Choose from:
- Onset (when it started)
- Location/where
- Duration/how long
- Character/description
- Severity
- Factors that make it better/worse
- Associated symptoms

Pick the most logical next question based on what's already been discussed, but don't repeat questions already answered.
"""

        # Medical history phase
        if topic == "medical_history":
            return "Ask about relevant medical history, including any chronic conditions or previous issues related to their current symptoms."
            
        # Medication phase
        if topic == "medications":
            return "Ask about current medications the patient is taking, including prescription medications, over-the-counter drugs, or supplements."
            
        # Allergies phase
        if topic == "allergies":
            return "Ask about any allergies the patient may have, including medication allergies, food allergies, or environmental allergies."
            
        # Goals phase
        if topic == "goals":
            return "Ask about the patient's main concerns or goals for this consultation."
            
        # Final wrap-up
        if exchange_count > 12:
            return "Assess if we have covered all necessary topics. If so, prepare to conclude the conversation."
            
        # Default - continue with current topic
        return f"Continue gathering information about the patient's {topic.replace('_', ' ')}."
    
    # Function to analyze conversation progress and move to next topic when appropriate
    def should_advance_topic(topic, exchange_count, last_exchanges):
        # Extract last 2 user messages and last 2 AI messages (if available)
        user_msgs = [msg for i, msg in enumerate(last_exchanges) if i % 2 == 0][-2:] if len(last_exchanges) >= 2 else []
        ai_msgs = [msg for i, msg in enumerate(last_exchanges) if i % 2 == 1][-2:] if len(last_exchanges) >= 3 else []
        
        # Convert to lowercase for easier analysis
        user_msgs = [msg.lower() for msg in user_msgs]
        ai_msgs = [msg.lower() for msg in ai_msgs]
        
        # Very brief responses might indicate we should move on
        if any(len(msg.split()) <= 3 for msg in user_msgs) and exchange_count > 3:
            return True
            
        # If we've spent enough time on a topic
        topic_exchanges = {
            "chief_complaint": 2,     # Move on after 2 exchanges about chief complaint
            "symptom_details": 6,     # Spend more time on symptoms
            "medical_history": 2,
            "medications": 2,
            "allergies": 1,
            "goals": 2
        }
        
        if exchange_count >= topic_exchanges.get(topic, 3):
            return True
            
        # Move on if AI seems to be repeating questions
        if len(ai_msgs) >= 2:
            ai_msg1 = ai_msgs[0]
            ai_msg2 = ai_msgs[1]
            
            # Check for similar questions being asked twice
            similarity = len(set(ai_msg1.split()) & set(ai_msg2.split())) / max(len(ai_msg1.split()), len(ai_msg2.split()))
            if similarity > 0.6 and "?" in ai_msg1 and "?" in ai_msg2:
                return True
                
        return False
    
    # Function to check if we have gathered sufficient information for all necessary areas
    def information_is_complete(conversation_history):
        # Extract messages for analysis
        all_text = " ".join(conversation_history).lower()
        user_msgs = [msg.lower() for i, msg in enumerate(conversation_history) if i % 2 == 0]
        
        # Critical information needed for most complaints
        critical_info = {
            "chief_complaint": False,  # What's the main issue?
            "duration": False,         # How long has it been happening?
            "severity": False,         # How bad is it?
            "character": False,        # What does it feel like?
            "factors": False,          # What makes it better/worse?
            "medical_history": False,  # Any relevant medical history?
            "medications": False,      # Taking any medications?
            "allergies": False,        # Any allergies?
            "goals": False,            # What does the patient want?
        }
        
        # Check for chief complaint
        complaint_indicators = ["headache", "pain", "fever", "cough", "rash", 
                               "nausea", "vomiting", "dizzy", "ache", "hurt"]
        if any(indicator in all_text for indicator in complaint_indicators):
            critical_info["chief_complaint"] = True
        
        # Check for duration information
        duration_indicators = ["hour", "day", "week", "month", "year", "minute", 
                              "since", "started", "ago", "begin", "onset"]
        if any(indicator in all_text for indicator in duration_indicators) and any(char.isdigit() for char in all_text):
            critical_info["duration"] = True
        
        # Check for severity information
        severity_indicators = ["scale", "rate", "severe", "mild", "moderate", "worse", 
                              "better", "intensity", "/10", "out of 10"]
        if any(indicator in all_text for indicator in severity_indicators) or any(f"{i}/10" in all_text or f"{i} out of 10" in all_text for i in range(1, 11)):
            critical_info["severity"] = True
        
        # Check for pain character information
        character_indicators = ["sharp", "dull", "aching", "throbbing", "burning", 
                               "stabbing", "pressure", "feels like", "sensation"]
        if any(indicator in all_text for indicator in character_indicators):
            critical_info["character"] = True
        
        # Check for aggravating/alleviating factors
        factor_indicators = ["worse", "better", "improves", "aggravates", "trigger", 
                            "alleviate", "worsen", "help", "make it", "reduces"]
        if any(indicator in all_text for indicator in factor_indicators):
            critical_info["factors"] = True
        
        # Check for medical history
        history_indicators = ["condition", "chronic", "diagnosed", "surgery", "hospital", 
                             "previous", "past", "medical history", "health condition"]
        # Also check for yes/no responses to medical history questions
        if any(indicator in all_text for indicator in history_indicators) or ("history" in all_text and any(neg in all_text for neg in ["no", "don't have", "none"])):
            critical_info["medical_history"] = True
        
        # Check for medication information
        med_indicators = ["medication", "medicine", "drug", "pills", "prescribe", 
                         "supplement", "vitamin", "over-the-counter"]
        # Also check for yes/no responses to medication questions
        if any(indicator in all_text for indicator in med_indicators) or ("medication" in all_text and any(neg in all_text for neg in ["no", "don't take", "none"])):
            critical_info["medications"] = True
        
        # Check for allergy information
        allergy_indicators = ["allergy", "allergic", "reaction", "sensitive", "anaphylaxis"]
        # Also check for yes/no responses to allergy questions
        if any(indicator in all_text for indicator in allergy_indicators) or ("allerg" in all_text and any(neg in all_text for neg in ["no", "don't have", "none"])):
            critical_info["allergies"] = True
        
        # Check for goals/concerns
        goal_indicators = ["want", "hope", "expect", "goal", "concern", "worry", 
                          "looking for", "help with", "improve", "relief", "need", "cure"]
        if any(indicator in all_text for indicator in goal_indicators):
            critical_info["goals"] = True
        
        # For the conversation to be complete, we need the complaint + majority of critical info
        completeness_percentage = sum(1 for info in critical_info.values() if info) / len(critical_info)
        
        # The chief complaint and at least 70% of critical info must be present
        return critical_info["chief_complaint"] and completeness_percentage >= 0.7
    
    # Function to generate a summary of collected information for final confirmation
    def generate_conversation_summary(conversation_history):
        # Prompt the AI to create a brief summary of what's been discussed
        summary_prompt = """
Create a brief summary (2-3 sentences) of the key information gathered from our conversation, including:
1. Chief complaint (what's bothering the patient)
2. Key symptoms (duration, severity, character, location)
3. Any relevant factors that make it better/worse
4. Medical history, medications, or allergies mentioned

Keep it concise and focused on facts stated by the patient.
"""
        # Generate the summary with lower temperature for consistency
        return generate_response(conversation_history, summary_prompt, temperature=0.4)
    
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nAI: Ending the consultation without creating a summary.")
            return
        
        if user_input.lower() == 'summary':
            print("\nAI: I'll generate a summary of our conversation now.")
            break
            
        # Add user input to conversation history
        conversation_history.append(user_input)
        
        # Check if we're in the closing confirmation phase
        in_closing_phase = len(conversation_history) >= 3 and "is there anything important i missed" in conversation_history[-2].lower()
        
        # Handle response to final confirmation
        if in_closing_phase:
            # Check if user indicated we're done with a brief negative response
            negative_responses = ["no", "nope", "nothing", "that's all", "all good", "that covers it", "that's it", "nah"]
            if any(resp in user_input.lower() for resp in negative_responses) or len(user_input.split()) <= 3:
                # User confirmed we're done - generate summary and exit
                print("\nAI: Thank you for confirming. I'll generate a summary of our conversation now.")
                break
            else:
                # User has more to add - acknowledge and continue conversation
                follow_up_prompt = "The patient has additional information to share. Acknowledge this new information and ask a relevant follow-up question based on what they've added."
                follow_up_response = generate_response(conversation_history, follow_up_prompt)
                print(f"\nAI: {follow_up_response}")
                conversation_history.append(follow_up_response)
                exchange_count += 1
                continue
        
        # Analyze progress and potentially advance topic
        if should_advance_topic(current_topic_area, exchange_count, conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history):
            # Find current topic index and advance if possible
            current_index = topic_areas.index(current_topic_area)
            if current_index < len(topic_areas) - 1:
                prev_topic = current_topic_area
                current_topic_area = topic_areas[current_index + 1]
                
                # Get transition text
                transition_key = f"{prev_topic}_to_{current_topic_area}"
                transition = topic_transitions.get(transition_key, f"Let's talk about your {current_topic_area.replace('_', ' ')}.")
                
                print(f"\n[DEBUG: Moving from {prev_topic} to {current_topic_area}]")
            else:
                # We've covered all topics - check information completeness before closing
                if information_is_complete(conversation_history):
                    # Generate a summary of the information collected so far
                    information_summary = generate_conversation_summary(conversation_history)
                    
                    # Create final confirmation prompt with the summary
                    final_prompt = f"Based on our conversation, I've gathered the following information: {information_summary} Is there anything important I missed or anything else you'd like to add?"
                    final_response = generate_response(conversation_history, final_prompt)
                    print(f"\nAI: {final_response}")
                    conversation_history.append(final_prompt)
                    conversation_history.append(final_response)
                    
                    # Add a flag to indicate we're in the closing phase
                    exchange_count += 1
                    continue
        
        # Get context-based prompt for the current topic
        context_prompt = get_context_prompt(current_topic_area, exchange_count, conversation_history)
        
        # Create the AI prompt with contextual guidance
        if exchange_count > 0:
            ai_prompt = f"Respond naturally to the patient. {context_prompt}"
        else:
            ai_prompt = context_prompt
        
        # Print current topic for debugging
        print(f"\n[DEBUG: Current topic area: {current_topic_area}]")
        
        # Generate response
        response = generate_response(conversation_history, ai_prompt)
        print(f"\nAI: {response}")
        
        # Add response to conversation history
        conversation_history.append(response)
        
        # Increment exchange counter
        exchange_count += 1
        
        # Check if we've had enough exchanges and have complete information
        if exchange_count >= 15 and information_is_complete(conversation_history):
            # Generate a summary of the information collected so far
            information_summary = generate_conversation_summary(conversation_history)
            
            # Create final confirmation prompt with the summary
            final_prompt = f"Based on our conversation, I've gathered the following information: {information_summary} Is there anything important I missed or anything else you'd like to add?"
            final_response = generate_response(conversation_history, final_prompt)
            print(f"\nAI: {final_response}")
            conversation_history.append(final_prompt)
            conversation_history.append(final_response)
        
        # We still have a maximum exchange count as a fallback, but it's higher
        if exchange_count >= 25:
            wrap_prompt = "Thank the patient and let them know we'll prepare a summary of the conversation for the doctor."
            wrap_response = generate_response(conversation_history, wrap_prompt)
            print(f"\nAI: {wrap_response}")
            conversation_history.append(wrap_prompt)
            conversation_history.append(wrap_response)
            break
    
    # Gather patient information for the summary
    print("\nTo complete the summary, please provide the following information:")
    patient_name = input("Patient Name: ") or "[Patient Name]"
    patient_id = input("Patient ID: ") or "[Patient ID]"
    
    # Generate SOAP note
    soap_text = generate_soap_summary(conversation_history, patient_name, patient_id)
    
    # Print the SOAP note
    print("\n===== PATIENT INTAKE SUMMARY =====")
    print(soap_text)
    print("==================================\n")
    
    # Generate filename based on patient name and date
    filename = f"SOAP_{patient_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    
    # Create and save the PDF
    pdf_path = create_soap_pdf(soap_text, filename)
    print(f"SOAP note saved as {pdf_path}")

if __name__ == "__main__":
    chat() 