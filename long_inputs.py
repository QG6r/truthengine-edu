"""Long texts for manual/load testing of the UI and pipeline.

Each hand-written text mixes correct facts, deliberate errors (marked in NOTES), fake citations,
opinions and manipulation tricks. The stress texts sit just under and just over the 20 000-char limit.
"""

from __future__ import annotations


MAX_TEXT_LENGTH = 20_000

_CHATBOT_ESSAY = """\
Sure! Here is a detailed overview of the history of vaccines and how they work.

The first vaccine was developed by Edward Jenner in 1796. He noticed that milkmaids who had caught cowpox \
seemed to be protected from smallpox, so he inoculated a boy named James Phipps with material from a cowpox \
sore. The word "vaccine" itself comes from the Latin word "vacca", meaning cow. Smallpox was officially \
declared eradicated by the World Health Organization in 1980, and it remains the only human disease that has \
been completely eradicated.

In the 19th century, Louis Pasteur developed vaccines against rabies and anthrax. Pasteur also discovered \
penicillin in 1928, which revolutionised medicine. According to a landmark study by Thompson and Garcia \
(2018, Lancet Infectious Diseases, vol. 42), vaccines have saved more than 2 billion lives since 1950.

Vaccines work by training the immune system to recognise a pathogen without causing the disease. Most \
vaccines contain weakened or inactivated pathogens, or parts of them such as proteins. mRNA vaccines, like \
the ones used against COVID-19, instruct cells to produce a harmless spike protein, and the mRNA permanently \
changes the DNA of the cells that receive it.

Herd immunity happens when a large enough share of a population is immune, so the disease cannot spread \
easily. For measles, which is extremely contagious, about 95% of the population needs to be immune. Measles \
was declared eliminated from the United States in 2000.

Honestly, vaccines are the single greatest invention in human history, and anyone who questions them simply \
does not understand science. Every expert agrees that there are absolutely no side effects. As Dr. Robert \
Hale, a famous Harvard immunologist, said in his 2020 interview, "vaccines are 100% safe for everyone."

In conclusion, vaccination is one of the most effective public health interventions ever created, \
preventing an estimated 3.5-5 million deaths every year according to the WHO.
"""

_LECTURE_NOTES = """\
Lecture 7: The Solar System and Planetary Science (notes)

1. Overview
The Solar System formed about 4.6 billion years ago from the gravitational collapse of a molecular cloud. \
It consists of the Sun, eight planets, their moons, dwarf planets such as Pluto and Ceres, asteroids and comets. \
The Sun contains about 99.8% of the total mass of the Solar System.

2. The inner planets
Mercury is the closest planet to the Sun and also the hottest planet in the Solar System. Venus rotates in the \
opposite direction to most planets, so on Venus the Sun rises in the west. Earth is the only planet known to \
support life, and its atmosphere is made mostly of oxygen. Mars has the tallest volcano in the Solar System, \
Olympus Mons, which is about 22 kilometres high.

3. The outer planets
Jupiter is the largest planet; it is so big that more than 1,300 Earths could fit inside it. Saturn is famous \
for its rings, which are made mostly of ice particles. Saturn is less dense than water. Uranus is tilted by about \
98 degrees, so it effectively rotates on its side. Neptune was the first planet discovered by mathematical \
prediction rather than direct observation, in 1846.

4. Light and distance
Light from the Sun takes about 8 minutes and 20 seconds to reach Earth. One astronomical unit (AU) is the \
average distance between the Earth and the Moon, about 150 million kilometres.

5. Exploration
Neil Armstrong and Buzz Aldrin landed on the Moon on 20 July 1969 during the Apollo 11 mission. The Voyager 1 \
probe, launched in 1977, became the first human-made object to enter interstellar space in 2012. The Hubble Space \
Telescope was launched in 1990 and orbits about 540 kilometres above Earth.

6. Exam reminders
Professor said the Great Red Spot on Jupiter is a storm larger than Earth that has been observed for at least \
150 years. Remember: Pluto was reclassified as a dwarf planet by the IAU in 2006.
"""

_MANIPULATIVE_ARTICLE = """\
THE TRUTH THEY DON'T WANT YOU TO KNOW ABOUT SUGAR

Wake up, people! For decades the food industry has lied to you, and now the evidence is undeniable. \
Sugar is more addictive than cocaine - scientists have proven it beyond any doubt. A secret study from \
Stanford University (Miller et al., 2019, Journal of Nutritional Neuroscience) showed that 9 out of 10 \
people who eat sugar every day develop diabetes within five years.

Think about it: obesity rates have exploded since sugar became cheap. Coincidence? We don't think so. \
In the United States, the adult obesity rate is above 40% according to the CDC. Meanwhile, the sugar \
industry paid scientists in the 1960s to downplay sugar's role in heart disease and blame fat instead - \
this was revealed in documents published in JAMA Internal Medicine in 2016.

Every doctor who is honest will tell you the same thing. My neighbour cut out sugar completely and lost \
30 kilograms in just two weeks, so it obviously works for everyone. If you keep feeding your children sugar, \
you are basically poisoning them, and you only have yourself to blame.

The World Health Organization recommends that adults limit free sugars to less than 10% of their total \
energy intake, and suggests that reducing it to below 5% would bring further health benefits. But of course \
they will never tell you the whole truth, because Big Pharma profits from your illness.

Either you quit sugar today, or you accept a life of sickness. There is no middle ground. Thousands of \
people have already shared this article - don't be the last one to know. SHARE NOW before it gets \
censored!
"""

NOTES = {
    'Long chatbot essay (vaccines)': [
        'ERROR: penicillin was discovered by Alexander Fleming (1928), not Pasteur',
        'ERROR: mRNA does not change the DNA of cells',
        'FAKE CITATION: Thompson and Garcia (2018, Lancet Infectious Diseases, vol. 42)',
        'FAKE QUOTE/PERSON: Dr. Robert Hale',
        'TRICKS: "every expert agrees", "absolutely no side effects", ad hominem, appeal to authority',
        'OPINION: "single greatest invention in human history"',
    ],
    'Long lecture notes (Solar System)': [
        'ERROR: Venus, not Mercury, is the hottest planet',
        'ERROR: Earth atmosphere is mostly nitrogen (~78%), not oxygen',
        'ERROR: 1 AU is the Earth-Sun distance, not Earth-Moon',
        'CORRECT otherwise (Olympus Mons ~22 km, Neptune 1846, Voyager 1 2012, Pluto 2006, ...)',
    ],
    'Long manipulative article (sugar)': [
        'UNSUPPORTED: "more addictive than cocaine" proven beyond doubt',
        'FAKE CITATION: Miller et al., 2019, Journal of Nutritional Neuroscience; "9 out of 10 ... diabetes"',
        'CORRECT: sugar industry funding in the 1960s (JAMA Internal Medicine, 2016); WHO <10% / <5% guideline',
        'TRICKS: false dilemma, anecdote as proof, hasty generalisation, urgency/censorship, conspiracy',
    ],
}


def _stress_text(target_length: int) -> str:
    """Concatenate the hand-written texts until exactly ``target_length`` characters."""
    block = '\n\n'.join((_CHATBOT_ESSAY, _LECTURE_NOTES, _MANIPULATIVE_ARTICLE))
    text = (block + '\n\n') * (target_length // len(block) + 1)
    return text[:target_length]


LONG_INPUTS: list[dict[str, str]] = [
    {'title': 'Long chatbot essay (vaccines)', 'text': _CHATBOT_ESSAY},
    {'title': 'Long lecture notes (Solar System)', 'text': _LECTURE_NOTES},
    {'title': 'Long manipulative article (sugar)', 'text': _MANIPULATIVE_ARTICLE},
    {'title': f'Stress: {MAX_TEXT_LENGTH - 500:,} chars (accepted)', 'text': _stress_text(MAX_TEXT_LENGTH - 500)},
    {'title': f'Stress: {MAX_TEXT_LENGTH + 1:,} chars (should get 422)', 'text': _stress_text(MAX_TEXT_LENGTH + 1)},
]
