# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
**VibeFinder 1.0**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

This model suggest 3-5 songs from a small cataglog based on a user's preferred genre, mood, and target energy level. It will assume the user has a single preference for their listening session. 

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

The system uses content based filtering. It will compare the songs based on the tags tied to the songs. 
- 1 point if the genre matches
- 1 point if the mood matches
- up to 2 points if the energy matches the users energy

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

The model is small and represents genres like pop, lofi, rock and edm


---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

The would work well if the user is well defined and likes our dataset. If the user likes lofi or pop they will recieve accurate recommendations because the dataset is built for them.

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

If a user wants high energy music but their favorite genre is ambient. The system will ignore their energy and force low enerfy ambient songs onto them. It also would give poor recommendations to fancs of global music genres.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

I evaluated the system by stress testing with 4 profiles that have different tastes.
I was somewhat at how the system will will follow the tag more than the energy for a user that wants ambient music but has high energy.
I halved the the genre weight and doubled the energy weight so it would not get confused with conflicted users who want energy but have distict favorite genres.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

I would allow the users to input multiple favorite genres instead of just one. I would also take the other numerical data into consideration.
I would also have profile history so it relys on what the user listens to instead of static properties.


---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  


This tought me that prioritizing different categories can significatly alter the output. It makes me wonder how systems like spotify recommend music, since they would have millions of songs to consider and potentially thousands of song properties.
