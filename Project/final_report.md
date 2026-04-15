# Report

### Summary
This project involves the training of a transformer model from scratch using a dataset of almost 300k English/Spanish sentence pairs. Although the frontend is local, it includes an interface similar to Google translate, but features a save option for selected translations.

### Diagrams
![alt text](drawSQL-image-export-2026-04-03.jpg)

![alt text](IMG_8118.jpeg)

### Demo Video
[![Watch demo](452_demo_thumbnail.png)](452_project_demo.mp4)

### Learning Outcomes
- I learned how document-oriented databases differ from relational ones. MongoDB stores each translation as a self-contained JSON-like document rather than a row in a table, which means there's no schema to define upfront. You just insert data and the structure is implied by the documents themselves.
- I learned how a database runs as an independent service that an application connects to rather than something the code "contains." MongoDB runs in the background on its own, and the Python script connects to it, uses it, and then disconnects. The data persists regardless of whether the app is running.
- I also learned the practical implementation of CRUD operations. Creating documents on save, reading them to display the list, and deleting them individually or in bulk. These are the building blocks of every database-driven application.

### How the Project Integrates with AI
The project is built around a custom transformer neural network, which is trained entirely from scratch on 271,000 English-Spanish sentence pairs. The model learned the statistical relationships between the two languages through 10 epochs of training on a T4 GPU. There are no pre-trained weights, no external translation APIs, and no borrowed models. The intelligence in the system came entirely from the training process on the singular provided dataset.

### Use of AI to Create Project
I wrote the training script and the execution script as rough drafts, using tools aquired through CS 474. Upon a first "successful" training, I consulted Claude to identify patterns where the model was translating incorrectly, and it gave me pointers on what variables to adjust. Claude also helped me understand the concepts involved with MongoDB since I missed several lectures.

### Personal Interest
Ever since I got a solid grip on Spanish in my mission, I grew more respect for my Lola for learning English as a third language. I also grew very curious about learning various foreign languages, including but not limited to Ilocano (her dialect from the Philippines). Upon discovering the very basic concept of machine learning when I arrived home, my first thought was to create my very own translator.

### Strategy and Performance
- The primary bottleneck is inference speed on CPU. The model runs locally on a MacBook Air without GPU acceleration, which means response time increases with sentence length.
- MongoDB data is written to disk and survives restarts. If the Flask server crashes it can be restarted with a single command and reconnects to MongoDB automatically. There is no replication, backup schedule, or redundancy. A hard drive failure would mean permanent data loss. For the model itself, failover means rerunning the training script on Colab.