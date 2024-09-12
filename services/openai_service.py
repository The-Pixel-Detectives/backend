from openai import OpenAI
from config import settings
from schemas import TranslationRespone


translation_prompt = '''
Translate the following query to English as an input for a video search engine (brief, keyword-focused). Each query must provide all context to search for the target frame independently using single-frame image search.

### Instructions
Step 1: Translate the query to English.
Step 2: Identify frames are described in the query.
Step 3: Write query for each frame.

### Examples
Input: Khung cảnh trong một phòng làm việc, có một cái bàn tròn ở giữa và một cái bàn hình cung tròn bao quanh. Nhiều người đang ngồi làm việc trước các màn hình lớn đặt trên các bàn cung tròn này. Bảng thông tin trên màn hình của một người có chữ "FLIGHT DYNAMICS" khi đọc ngược. Tiếp sau đó là cảnh rất nhiều người vỗ tay ăn mừng.
Number of frames: 2

Step 1: Translation
The scene is in an office space, with a round table in the center and a curved table surrounding it. Many people are sitting and working in front of large screens placed on these curved tables. The information board on one person’s screen shows the words 'FLIGHT DYNAMICS' when viewed in reverse. Following that is a scene where many people are clapping and celebrating.

Step 2: There are 2 frames being describe:
1. Scene of an office with a round table in the middle and a semicircular desk surrounding it. Multiple people working in front of large screens on the semicircular desks. Information screen showing the text "FLIGHT DYNAMICS" in reverse.
2. In the same room, people clapping and celebrating.

Step 3: Write query for each frame
### Step 1: Identify how many frames are described:
1. Scene of an office with a round table in the middle and a semicircular desk surrounding it.
2. Multiple people working in front of large screens on the semicircular desks.
3. Information screen showing the text "FLIGHT DYNAMICS" in reverse.
4. People clapping and celebrating.

### Step 2: Write query for each frame:
**Frame 1:**
An office space, with a round table in the center and many people are sitting and working in front of large screens placed on these curved tables. The information board on one shows the words 'FLIGHT DYNAMICS' in reverse.

**Frame 2:**
An office space, with a round table in the center and many people are standing up and clapping in front of large screens placed on these curved tables.

Now, let's begin!

Input: {query}
Number of frames: {num_frames}
'''

class _OpenAIService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY
        )

    def translate_query(self, text: str, num_frames: int):
        '''
        text: original query in Vietnamese
        '''
        num_frames = max(num_frames, 1)
        response = self.client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": translation_prompt.format(
                query=text,
                num_frames=num_frames
            )}
          ]
        )

        response = response.choices[0].message.content
        print(response)

        completion = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract the frame descriptions from step 3 into a list of string. Each frame description is 1 string. Make sure the number of strings in the list equals the number of frames."},
                {"role": "user", "content": response},
            ],
            response_format=TranslationRespone,
        )

        return completion.choices[0].message.parsed


OpenAIService: _OpenAIService = _OpenAIService()
