# Telemedicine AI Intake Assistant

An AI-powered medical chatbot designed for telemedicine patient intake and automatic SOAP note generation.

## Features

* **Intelligent Medical Intake**: Conducts a professional, structured medical interview to collect patient information before telemedicine consultations
* **OLDCARTS Framework**: Thoroughly gathers History of Present Illness (HPI) details following standard medical protocol:
  * **O**nset: When symptoms began
  * **L**ocation: Where symptoms are experienced
  * **D**uration: How long symptoms last
  * **C**haracter: Quality/description of symptoms
  * **A**ggravating/Alleviating Factors: What makes symptoms better or worse
  * **R**adiation: Whether symptoms spread to other areas
  * **T**iming: Pattern of symptom occurrence
  * **S**everity: Intensity rating of symptoms
* **Emergency Detection**: Identifies potentially critical symptoms and provides urgent care guidance
* **Comprehensive History Collection**: Systematically gathers past medical history, current medications, and allergies
* **SOAP Note Generation**: Automatically creates the Subjective (S) section of SOAP notes with templates for clinical completion
* **PDF Export**: Generates professionally formatted PDF documents with complete SOAP structure

## Project Structure

```
├── telemedicine_intake_pdf.py       # AI-powered medical intake system with PDF generation
├── requirements.txt                 # Package dependencies
├── README.md                        # This documentation file
└── SOAP_*.pdf                       # Generated SOAP notes from patient interactions
```

## Requirements

* Python 3.8+
* torch (PyTorch)
* transformers
* reportlab (for PDF generation)
* NVIDIA GPU with CUDA support (optional, for faster processing)

The system will use GPU acceleration if available, otherwise it will fall back to CPU.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd telemedicine-chatbot
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Verify CUDA availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

```bash
python telemedicine_intake_pdf.py
```

## Interaction Guide

During the conversation:
* Answer the AI's questions about your symptoms and medical history
* The AI will guide you through a complete medical intake process
* Type `summary` at any point to generate a SOAP note based on the conversation
* Type `exit` to end the conversation without generating a summary

For PDF generation:
* After typing `summary`, you'll be prompted to provide patient name and ID
* A PDF file with the SOAP note will be automatically generated in the format `SOAP_{patient_name}_{timestamp}.pdf`
* The PDF includes templates for the doctor to complete during consultation

## Conversation Flow

The AI follows a clinically structured conversation flow:

1. **Initiation & Chief Complaint (CC)**: Identifies the main reason for the visit
2. **History of Present Illness (HPI)**: Explores symptom details using the OLDCARTS framework
3. **Past Medical History (PMH)**: Records relevant ongoing health conditions and past surgeries
4. **Current Medications**: Documents all medications, including prescriptions, OTC drugs, and supplements
5. **Allergies**: Notes medication, food, and environmental allergies
6. **Patient Goals/Concerns**: Captures specific patient worries or goals for the visit

## Intelligent Conversation Management

The system uses sophisticated tracking to ensure comprehensive information gathering:

* **Context-Aware Questioning**: Adapts questions based on previous responses
* **Natural Progression**: Smoothly transitions between medical topics
* **Complete Coverage**: Ensures all necessary medical information is collected
* **Emergency Protocol**: Detects critical symptoms and provides appropriate guidance

## SOAP Note Generation

The system generates structured SOAP notes following medical standards:

* **Subjective Section**: Automatically compiled from the conversation
* **Template Sections**: Includes professional templates for Objective, Assessment, and Plan
* **PDF Formatting**: Creates clean, organized documents with proper medical headings
* **Clinical Workflow Integration**: Designed to seamlessly fit into telemedicine workflows

## Model Information

The chatbot is powered by the Qwen2.5-3B-instruct-medical-finetuned model:
* 3 billion parameter model specifically fine-tuned for medical conversations
* Optimized for accurate, empathetic healthcare communication
* The model will use CUDA acceleration if available, otherwise falls back to CPU

## Safety Features

* **No Medical Advice**: The AI strictly avoids providing medical advice, diagnosis, or treatment suggestions
* **Emergency Detection**: Identifies potentially serious symptoms and provides guidance to seek immediate care
* **Clear Documentation**: Indicates AI-generated content vs. sections requiring clinical input
* **Factual Reporting**: Reports only information explicitly stated by the patient 