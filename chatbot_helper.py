from google import genai


def get_chatbot_response(api_key, user_message, data):

    try:

        client = genai.Client(api_key=api_key)

        airline_summary = {}

        for airline in data["airline"].unique():

            airline_data = data[data["airline"] == airline]

            airline_summary[airline] = {
                "total_tweets": len(airline_data),
                "sentiments":
                airline_data["airline_sentiment"]
                .value_counts()
                .to_dict()
            }

        top_negative_reasons = (
            data["negativereason"]
            .dropna()
            .value_counts()
            .head(10)
            .to_dict()
        )

        prompt = f"""
        You are an Airline Sentiment Analysis Assistant.

        Dataset Information:

        Total Tweets:
        {len(data)}

        Airline Summary:
        {airline_summary}

        Top Negative Reasons:
        {top_negative_reasons}

        Rules:
        - Answer only from dataset information.
        - Keep answers concise.
        - Compare airlines when asked.
        - If asked for best airline, use positive vs negative sentiment.
        - Mention negative reasons when relevant.
        - Use simple language

        User Question:
        {user_message}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:
        return str(e)
