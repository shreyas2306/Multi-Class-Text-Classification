# Mullti-Class-Text-Classification


Api service which aims to classify a text into a given topic using machine learning algorithms.

Following topics are listed below:

- Animals: Statements talking about animals
- Compliment: Mostly positive, encouraging statements
- Education: Centers around education and their policies, schooling, etc.
- Health: News about health
- Heavy Emotion: Statements that convey mostly negative or strong emotion, usually anger
- Joke: Statements that are part of a joke
- Love: Romantic statements, experiences, or about love itself
- Politics: Everything going on in governments worldwide, mostly U.S.
- Religion: Discussion about all kinds of religion
- Science: Various statements concerning discoveries or research in science
- Self: Statements where the speaker is talking about themselves

## Basic Requirements
- **Python3.0+**

### Dependencies:

- pandas>=0.25.1
- numpy>=1.17.3
- transformers==4.11.3
- nltk==3.5
- flask==1.1.2
- Flask-RESTful==1.1.1


### Python version
    `Python 3.8.3`

### Installation

Install all the dependencies using the following command.

`pip install -r requirements.txt`

then run the following command:

`python main.py <input statement in json>`

e.g. python main.py {"statement": "While lifting weights on Friday and doing bent over rows, I felt a sharp pain in my lower back. Some would say I threw it out and I had this happen one other time about 6 months ago. Dropped to my knees and was sore for the rest of the day. Overnight, it just became terrible. Every time my body wanted to move, I woke everybody in the room up with moans of pain (wife, dog, cats)."}

### Output

{"statement": "While lifting weights on Friday and doing bent over rows, I felt a sharp pain in my lower back. Some would say I threw it out and I had this happen one other time about 6 months ago. Dropped to my knees and was sore for the rest of the day. Overnight, it just became terrible. Every time my body wanted to move, I woke everybody in the room up with moans of pain (wife, dog, cats).",
"topic":'joke'}
