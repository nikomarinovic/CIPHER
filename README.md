# CIPHER
Cognitive Intelligence &amp; Process Handling Engine for Response

---

Da. Ovo što već imaš je dobar početak, samo bih ga sada posložio tako da CIPHER može rasti u pravi AI agent, a ne da sve završi u core.py.

Ne bih još trenirao model. Prvo napravite da CIPHER stvarno zna koristiti alate, memoriju, internet i kontekst. Tek onda ima smisla razmišljati o vlastitom fine-tuningu/treningu.

Konačna struktura za sada

CIPHER/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   └── main.py
│
├── brain/
│   │
│   ├── __init__.py
│   ├── core.py
│   ├── router.py
│   ├── reasoning.py
│   ├── context.py
│   ├── memory.py
│   ├── internet.py
│   ├── config.py
│   │
│   ├── data/
│   │   ├── memory.db
│   │   └── knowledge.db
│   │
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── coding.py
│   │   ├── lookup.py
│   │   ├── conversation.py
│   │   └── research.py
│   │
│   └── tools/
│       ├── __init__.py
│       ├── calculator.py
│       ├── code_runner.py
│       ├── system.py
│       ├── web_search.py
│       └── file_manager.py
│
├── prompts/
│   ├── system.txt
│   ├── reasoning.txt
│   └── coding.txt
│
├── tests/
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_tools.py
│   └── test_skills.py
│
└── training/
    ├── datasets/
    └── README.md

Što je što

app/main.py

Samo pokretanje aplikacije i user input/output.

User
 ↓
main.py
 ↓
CIPHER

Nemoj ovdje stavljati AI logiku.

⸻

brain/core.py

Glavni mozak.

input
 ↓
context
 ↓
router
 ↓
reasoning
 ↓
tool/skill
 ↓
response

On koordinira sve ostalo.

⸻

brain/router.py

Ovo će vam biti jedan od najvažnijih fileova.

Njegov posao nije da odgovara korisniku.

Njegov posao je odlučiti:

“Što CIPHER treba napraviti s ovom porukom?”

Primjer:

"koliko je 928 * 72?"
→ calculator
"tko je trenutno CEO NVIDIA-e?"
→ web_search
"napravi Python funkciju za sortiranje"
→ coding
"sjećaš se kako se zove moj projekt?"
→ memory
"objasni mi pointere"
→ conversation/reasoning

Ali nemojte ovo napraviti kao 500 if naredbi.

Cilj je da router koristi AI/model da razumije namjeru.

⸻

brain/reasoning.py

Ovdje CIPHER odlučuje kako riješiti problem.

Na primjer:

User:
"Nađi mi najjeftiniji RTX 5070 u Hrvatskoj."
Reasoning:
1. User wants current information.
2. Need internet.
3. Search Croatian stores.
4. Compare prices.
5. Return result.

To je ono što će ga početi činiti više “agentom”.

⸻

brain/context.py

Drži trenutni razgovor.

Npr.:

User:
"Koliko RAM-a treba za SolidWorks?"
CIPHER:
"16 GB je dovoljno..."
User:
"A za velike assemblyje?"

CIPHER mora znati da se “za velike assemblyje” odnosi na SolidWorks/RAM.

⸻

brain/memory.py

Dugoročna memorija.

Npr.:

User:
"Zapamti da moj projekt koristi Python."
→ memory

Kasnije:

User:
"Kako da dodam novi modul?"
CIPHER:
"Pošto koristiš Python..."

To nije isto kao context.py.

context = trenutni razgovor

memory = stvari koje želiš pamtiti kroz razgovore

⸻

brain/internet.py

Visok nivo internet funkcionalnosti.

Ne bi trebao sadržavati ogromnu logiku.

On može koristiti:

tools/web_search.py

⸻

Skills vs Tools

Ovo je jako bitno.

Tools

Tools su konkretne sposobnosti.

calculator
web_search
code_runner
file_manager
system

Primjer:

calculator.calculate("928 * 72")

⸻

Skills

Skills koriste jedan ili više toolova da naprave nešto kompleksnije.

Primjer:

research.py
web_search
    ↓
multiple searches
    ↓
collect information
    ↓
reasoning
    ↓
answer

Zato je:

tools/web_search.py

alat.

A:

skills/research.py

sposobnost.

⸻

prompts/

Ovo bih vam obavezno dodao.

Nemojte system prompt zakucati direktno u Python.

system.txt može sadržavati osobnost i pravila CIPHER-a.

Primjer koncepta:

You are CIPHER.
You are an AI assistant designed to reason, use tools,
remember relevant information and research the internet.
Do not fabricate information.
When information may be outdated, use the web.
Do not perform unnecessary searches.
Prefer solving problems yourself when possible.
...

Kasnije ćete tu moći jako puno eksperimentirati.

⸻

training/

Ovo za sada ne dirajte ozbiljno.

Ali napravite ga odmah jer ćete kasnije tamo moći imati:

training/
├── datasets/
│   ├── conversations.jsonl
│   ├── tool_calls.jsonl
│   └── reasoning.jsonl
│
└── README.md

I možete skupljati primjere kako želite da CIPHER odgovara.

⸻

A gdje je zapravo AI model?

To vam trenutno nedostaje kao zasebna komponenta.

Ja bih dodao:

brain/
└── model.py

Dakle:

brain/
├── core.py
├── model.py
├── router.py
├── reasoning.py
...

model.py je jedino mjesto koje komunicira s vašim AI modelom/API-jem.

To je jako korisno jer kasnije možete promijeniti model bez mijenjanja cijelog CIPHER-a.

CIPHER
   │
   └── model.py
          │
          ├── Model A
          │
          ├── Model B
          │
          └── vaš fine-tuned model

⸻

Odakle krenuti?

Nemojte sada pokušavati napraviti sve odjednom.

Idite ovim redom:

1. model.py

Prvo napravite da CIPHER može:

user → model → response

Bez interneta, bez memoryja, bez toolova.

Samo da razgovara.

⸻

2. core.py

Napravite:

User
 ↓
CIPHER Core
 ↓
Model
 ↓
Response

⸻

3. context.py

Dodajte povijest razgovora:

User
 ↓
context
 ↓
model
 ↓
response

Tada već imate pravi chatbot.

⸻

4. router.py

Tek onda dodajte:

User
 ↓
Router
 ↓
┌──────────────┐
│ conversation │
│ calculator   │
│ web          │
│ coding       │
│ memory       │
└──────────────┘

⸻

5. Tools

Dodajete jedan po jedan:

calculator
↓
web search
↓
code runner
↓
file manager

Nakon svakog napravite test.

⸻

6. Memory

Tek kad razgovor i routing rade kako treba:

conversation
      +
memory

⸻

7. Research

Onda napravite research.py.

To je trenutak kada CIPHER počinje izgledati puno ozbiljnije:

"istraži mi X"
       ↓
CIPHER odluči:
"trebam internet"
       ↓
pretraži
       ↓
prikupi rezultate
       ↓
analizira
       ↓
sastavi odgovor

⸻

8. Tek nakon toga training

Kad imate stotine ili tisuće stvarnih interakcija, možete gledati:

User input
      ↓
CIPHER reasoning
      ↓
tool choice
      ↓
result
      ↓
ideal response

I od toga graditi dataset.

Tada vaš “trening” neće biti nasumičan — imat ćete konkretne primjere kako želite da se CIPHER ponaša.

⸻

I najvažnije za vas dvojicu

Nemojte obojica uređivati main.py i core.py istovremeno.

Podijelite module.

Npr.:

Ti:

core.py
router.py
reasoning.py
context.py
memory.py

Frend:

internet.py
tools/
skills/

I svaki radi:

git switch main
git pull --rebase origin main
git switch -c feature/router

radi svoj posao → commit → push → PR → review → merge.

Tako ćete imati pravi workflow kao mali software tim.

Prvi konkretni korak za CIPHER sada bih napravio model.py + core.py, pa tek onda router. To vam daje temelj na kojem se sve ostalo može graditi.