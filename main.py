import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from categories import categories
from responses import diet_plans

def determine_category(user_input):
    max_similarity = 0
    matched_category = None
    vectorizer = TfidfVectorizer()
    user_input_vector = vectorizer.fit_transform([user_input])
    for category, keywords in categories.items():
        category_keywords = ' '.join(keywords)
        category_vector = vectorizer.transform([category_keywords])
        similarity_score = cosine_similarity(user_input_vector, category_vector)[0][0]
        if similarity_score > max_similarity:
            max_similarity = similarity_score
            matched_category = category
    return matched_category

def get_random_diet_plan(category):
    return random.choice(diet_plans[category])

def main():
    user_prompt = input("Please enter your prompt: ")
    category = determine_category(user_prompt)
    if category:
        print(f"Category: {category}")
        diet_plan = get_random_diet_plan(category)
        print("Random Diet Plan:")
        print(diet_plan)
    else:
        print("Sorry, could not determine the category.")

if __name__ == "__main__":
    main()
