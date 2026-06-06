from google import genai


def get_chatbot_response(api_key, user_message, data):
    try:
        client = genai.Client(api_key=api_key)

        total_tweets = len(data)
        sentiment_counts = data["airline_sentiment"].value_counts().to_dict()
        airlines = ", ".join(data["airline"].unique())

        prompt = f"""
        You are an AI assistant for airline sentiment analysis.

        Rules:
        - Answer only using this dataset
        - Be very concise
        - Be helpful
        - Compare airlines if asked

        Dataset information:
        Total tweets: {total_tweets}
        Sentiment counts: {sentiment_counts}
        Airlines: {airlines}

        User question:
        {user_message}
        """

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return f"Error: {e}"