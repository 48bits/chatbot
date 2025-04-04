# Telemedicine SOAP Intake Chatbot

A medical chatbot designed for telemedicine patient intake and automatic SOAP note generation, with both GPU-accelerated and API-based versions.

## Features

* **Structured Medical Intake**: Follows a professional medical conversation flow to collect patient information before their telemedicine consultation
* **OLDCARTS Framework**: Systematically gathers History of Present Illness (HPI) following the standard medical approach
* **Emergency Detection Protocol**: Identifies potentially critical symptoms and provides emergency guidance
* **Comprehensive Patient History**: Collects past medical history, medications, and allergies
* **SOAP Note Generation**: Automatically generates the Subjective (S) section of a SOAP note based on the conversation
* **PDF Export**: Creates professionally formatted PDF documents with the SOAP note that includes templates for clinician completion
* **Dual Implementation**:
  * GPU-accelerated version for systems with compatible NVIDIA GPUs
  * API-based version that works on any system without special hardware requirements

## Project Structure

```
├── telemedicine_intake.py           # GPU-accelerated version of the chatbot
├── telemedicine_intake_pdf.py       # GPU-accelerated version with PDF generation
├── telemedicine_intake_api.py       # API-based version for systems without GPU
├── telemedicine_intake_api_pdf.py   # API-based version with PDF generation
├── check_cuda.py                    # Utility to check CUDA availability
└── README.md                        # This documentation file
```

## Requirements

### GPU Version
* Python 3.8+
* PyTorch with CUDA support
* transformers
* NVIDIA GPU with CUDA support (minimum 8GB VRAM recommended)
* reportlab (for PDF generation)

### API Version
* Python 3.8+
* requests
* json
* No specialized hardware requirements
* reportlab (for PDF generation)

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd telemedicine-chatbot
```

2. Install required packages:
```
pip install torch transformers requests reportlab
```

3. Verify CUDA availability (for GPU version):
```
python check_cuda.py
```

4. If using the API version, update the `HF_API_KEY` variable in `telemedicine_intake_api.py` or `telemedicine_intake_api_pdf.py` with your Hugging Face API key.

## Usage

### GPU-Accelerated Version
```
python telemedicine_intake.py
```

### GPU-Accelerated Version with PDF Generation
```
python telemedicine_intake_pdf.py
```

### API Version (No GPU Required)
```
python telemedicine_intake_api.py
```

### API Version with PDF Generation
```
python telemedicine_intake_api_pdf.py
```

## Interaction Commands

* Type your symptoms and medical history when prompted
* Type `summary` at any point to generate a SOAP note summary based on the conversation
* For PDF-enabled versions, you can save the SOAP note as a PDF file when generating a summary
* Type `exit` to end the conversation

## Preset Logic

The chatbot follows a specific conversation flow:

1. Gathers the Chief Complaint (CC)
2. Explores the History of Present Illness (HPI) using the OLDCARTS framework:
   * Onset: When the symptoms began
   * Location: Where the symptoms are felt
   * Duration: How long symptoms last
   * Character: Quality of the symptoms
   * Aggravating/Alleviating Factors: What makes symptoms better or worse
   * Radiation: Whether symptoms spread to other areas
   * Timing: When symptoms occur
   * Severity: How intense the symptoms are
3. Collects Past Medical History (PMH)
4. Records Current Medications
5. Documents Allergies
6. Notes Patient Worries/Goals for the visit

The chatbot includes an emergency protocol that detects potentially critical symptoms and advises immediate medical attention.

## Intelligent Conversation Management

The system uses a structured state-tracking approach for guiding the conversation:

### Explicit Section Tracking
* Tracks 6 required medical intake sections (CC, HPI, PMH, Medications, Allergies, Goals)
* Uses extensive keyword lists for each section to detect when they're being discussed

### Heuristic Matching
* Analyzes each exchange for relevant keywords to determine covered topics
* Special handling for Chief Complaint (detected early, or if user provides detailed first response)
* Special counter for HPI that requires multiple mentions (at least 3) due to its complexity

### Guided Conversation Flow
* Uses directive prompts for each missing section
* Checks every 3 exchanges for missing sections in a preferred order (CC → HPI → PMH → Meds → Allergies → Goals)
* When a section is missing, injects a directive prompt to guide the conversation

### Natural Completion
* Conversation automatically ends when all sections are marked as covered
* User can still type 'summary' or 'exit' at any point
* Maintains a 25-exchange maximum failsafe

This approach provides predictable control flow and ensures all necessary medical information is gathered without relying on the AI to assess completeness. It also provides natural guidance through the intake process in a structured manner.

## PDF Generation

When you type `summary`, the chatbot will:

1. Generate the Subjective (S) section of the SOAP note based on the conversation
2. Include templates for the clinician to complete the Objective (O), Assessment (A), and Plan (P) sections
3. Offer to save the document as a professionally formatted PDF file
4. Allow you to customize the patient name and ID for the report

The generated PDF includes:
* Patient information and consultation details
* Complete SOAP note structure
* Properly formatted sections with medical headings
* Visual indicators for areas requiring clinician input

## Model Information

The chatbot uses the Qwen2.5-3B-instruct-medical-finetuned model:
* 3 billion parameter model fine-tuned for medical conversations
* Hosted on Hugging Face by Joker-sxj
* Used locally in the GPU version and via API in the non-GPU version 