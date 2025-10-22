from groq import Groq
import json
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

def generate_quiz_questions(topic, num_questions=20,difficulty="mixed"):
    """
    Generate true/false quiz questions about a given topic using Groq.
    
    Args:
        topic (str): The topic for the quiz questions
        num_questions (int): Number of questions to generate (default: 20)
    
    Returns:
        list: List of dictionaries containing questions and answers
    """
    # Set your Groq API key from .env file
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        print("Error: Please set your GROQ_API_KEY in the .env file")
        print("You can get an API key from: https://console.groq.com/keys")
        return None
    
    client = Groq(api_key=api_key)
    
    # Create the prompt for Groq
    prompt = f"""Generate exactly {num_questions} true/false quiz questions about {topic} that are of difficulty {difficulty}.

Format your response as a JSON array where each question is an object with the following structure:
{{
    "question": "The question text",
    "answer": true or false,
    "explanation": "Brief explanation of why the answer is correct"
}}

Make the questions varied in difficulty and cover different aspects of {topic}.
Ensure a good mix of true and false answers.
Return ONLY the JSON array, no additional text."""

    try:
        # Call the Groq API
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # GPT OSS 120B model
            messages=[
                {"role": "system", "content": "You are a helpful quiz generator that creates educational true/false questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the response content
        content = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        questions = json.loads(content)
        
        return questions
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Response was: {content}")
        return None
    except Exception as e:
        print(f"Error generating questions: {e}")
        return None


def display_quiz(questions):
    """Display the quiz questions in a formatted way."""
    if not questions:
        return
    
    print("\n" + "="*80)
    print(f"QUIZ: {len(questions)} True/False Questions")
    print("="*80 + "\n")
    
    for i, q in enumerate(questions, 1):
        print(f"Question {i}:")
        print(f"  {q['question']}")
        print(f"  Answer: {q['answer']}")
        print(f"  Explanation: {q['explanation']}")
        print()


def save_quiz_to_file(questions, topic, filename=None):
    """Save the quiz questions to a JSON file."""
    if not questions:
        return
    
    if filename is None:
        # Create a safe filename from the topic
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        filename = f"quiz_{safe_topic}.json"
    
    quiz_data = {
        "topic": topic,
        "num_questions": len(questions),
        "questions": questions
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(quiz_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nQuiz saved to: {filename}")


def main():
    """Main function to run the quiz generator."""
    print("=" * 80)
    print("Groq Quiz Generator - True/False Questions")
    print("=" * 80)
    
    # Get topic from user
    topic = input("\nEnter the topic for your quiz: ").strip()
    
    if not topic:
        print("Error: Topic cannot be empty!")
        return
    
    print(f"\nGenerating 20 true/false questions about '{topic}'...")
    print("This may take a few moments...\n")
    
    # Generate questions
    questions = generate_quiz_questions(topic)
    
    if questions:
        # Display the questions
        display_quiz(questions)
        
        # Ask if user wants to save
        save = input("Would you like to save this quiz to a file? (y/n): ").strip().lower()
        if save == 'y':
            save_quiz_to_file(questions, topic)
        
        print(f"\nSuccessfully generated {len(questions)} questions!")
    else:
        print("Failed to generate quiz questions. Please check your API key and try again.")


if __name__ == "__main__":
    main()
