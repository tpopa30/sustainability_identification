from openai import OpenAI
import pandas as pd
from bs4 import BeautifulSoup
import json

client = OpenAI(
  api_key="YOUR_API_KEY_HERE"
)

zero_prompt = """
Input: You will receive a fragment of text which represents Task-Data, which will be a text fragment that you will receive as a user message.

Command: I want you to label the Task-Data with "Yes" if the text is related to software sustainability and with "No" in case the text fragment is unrelated". Besides labelling the post, you also need to provide a justification for the label given to the piece of text, i.e. explain why the text piece is related or not to sustainability.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "label": "<Yes/No>",
 "justification": "<Your justification>"
}

"""

definition_prompt = """

Input: You will receive two kinds of text: Helper-Data and Task-Data. The Task-Data is a text fragment that you will receive as a user message. The Helper-Data is comprised of text fragments that define software sustainability.

The following two definitions represent the Helper-Data: 
First definition: "Sustainability has multiple dimensions and here is a definition for each of them in the software intensive systems. The following paragraphs describe the dimensions used in the framework to characterize sustainability in the context of software-intensive systems: Social sustainability. Social sustainability focuses on ensuring current and future generations have the same or greater access to social resources by pursuing generational equity. For software-intensive systems, it encompasses the direct support of social communities in any domain, as well as activities or processes that indirectly create benefits for social communities; Environmental sustainability. Environmental sustainability aims to improve human welfare while protecting natural resources; for software-intensive systems, this means addressing ecological requirements, including energy efficiency and creation of ecological awareness; and Technical sustainability. Technical sustainability addresses the long-term use of software-intensive systems and their appropriate evolution in a constantly changing execution environment; and Economic sustainability. Economic sustainability focuses on preserving capital and financial value.\n An evaluation criterion can be a quality requirement. "

Second definition: " This definition contains sustainability across its dimensions, but not specifically for software systems. The economic dimension focuses on assets, capital and added value that comprises wealth creation, prosperity, profitability, capital investment, income, etc. The environmental dimension is concerned with the long-term effects of human activities on natural systems, which includes natural ecosystems and resources, the climate, pollution and waste, etc. The social dimension covers societal communities (groups of people, organisations) and the factors that erode trust in society. The concepts analysed here encompass social equity, justice, employment, democracy, etc. The technical dimension includes the concept of the longevity of information, systems, and infrastructure and their adequate evolution within changing environmental conditions, which covers inter alia, system maintenance, obsolescence, and data integrity"

Command: With the use of the Helper-Data you will have to label the Task-Data. I want you to label the Task-Data with "Yes" if the text is related to software sustainability and with "No" in case the text fragment is unrelated". Besides labelling the post, you also need to provide a justification for the label given to the piece of text, i.e. explain why the text piece is related or not to sustainability.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "label": "<Yes/No>",
 "justification": "<Your justification>"
}

"""

fewshot_prompt = """

Input: You will receive two kinds of text: Helper-Data and Task-Data. The Task-Data is a text fragment that you will receive as a user message. The Helper-Data is represented by examples of text and their label in order to help with your labelling.

The following text represents the Helper-Data in JSON Format: 

[
 {
 "text": "I am asking myself the exact same question at the moment.I am leaning towards the multi-instance single tenancy solution but have not taken a definitive decision yet. Let me share some of my thoughts: The main historical advantage of the multi-tenant architecture is a better use of infrastructure resources, by mutualisation (single OS, single Database, single application layer) and better occupying of said resources (when one user is away another can use the same resource). It also greatly simplifies software lifecycle : you deploy new versions to you one instance, all the customers are updated at the same time. It seems however, that recent advancements in cloud technology make the first class of advantages largely available in a multi-instance (instance-per-customer) architecture (I am thinking specifically of a platform like Jelastic here but I am sure there are others that provide the same features):\n\n • Container-based PaaS\n\n • Provisioning and auto-scaling of containers (elastic containers) \n\n So hardware and platform management is not the Software provider's concern any more. Resources are mutualised much more efficiently than before at the infrastructure and plaform levels. \n\n There will still be an overhead for multi-instance (some app and middleware will be ran N times instead of just one), but much lower than when using a separate (virtual) machine per instance. The database could be shared anyway (one schema per instance, several schemas per DB server)\nAlso :\n\n• Automation of creation of new instances is possible via PaaS API\n\n• Automation of deployment of new versions is possible via PaaS API, with zero downtime (takes some work to put in place)\n\n• Scaling is always out, never up. We don't have to worry about huge datasets at the instance level.\n\n Of course, we would need some kind of central service that manages all this automatically (e.g. creation of instance when a new user creates an account). This would also manage payment and licensing issues, interaction between instances etc. This central service might be quite complex and hard to develop, but the good thing is that we don't have to implement it upfront (now that we don't have much resource) whereas multi-tenant would need to be baked into the app from the start.\n\n Which brings me to the final advantages of developing single-tenant for a very early stage (pre-invesment) startup project :\n\n• Same (or almost same) version of the app can be deployed on-premises either as a virtual appliance or docker container or even on customer-managed machine (some companies are still reluctant towards the Cloud and it may help an early stage startup to not push out important early adopters)\n\n • Faster to get a product out with limited resources (the application layer and database schema is quite less complex), can get a 'dumb' single instance single tenant product out first (MVP) for early adopters and to show the business value of the app to potential investors, and add all the cloud automation later on \n\n• Can be seen as a selling argument for customers worried about data security : the data is better encapsulated since every customer has his own schema or even database. Much less risk of 'spillage'\n\nNB: I am obviously thinking here of a business app where customers would be businesses (each with multiple individual users) and not individuals. It would not make any sense to run a separate instance of an app for each individual user (or would it ?)",
 "label": "Yes"
 },
 {
 "text": "Node is a javascript execution environment. Node can act as a server (by executing a script that performs server functions).\nHeroku is an operating system (sitting in the cloud) that supports (among others) Node scripts. You can think of it like your desktop sitting on the internet, running a Node environment.\n When you run a server type of script in Heroku, it isn't any different than you running it on one your companies public facing servers.\n One of the nice things about Heroku over Apache, is that Heroku is designed to allow users to upload applications without an admin being required to deploy it.",
 "label": "No"
 },
 {
 "text": "We currently have a .Net MVC web app with a SQL server back-end database and two web services that perform periodically some computing tasks in the database.\n I am thinking of packaging this app for the cloud, where a client would 'instantiate' it as a whole package. This process would allocate automatically the pieces needed to run the whole thing, i.e. the site, the database and the other two services.\n I've been reading quite a bit about azure and I am having a very hard time getting my head around the whole azure development, testing and deployment process, then upgrading and applying changes to existing deployments. I am used to having control of all the pieces. First I like to have everything local on my workstation. We use git and we have our own self managed repository. I find it weird to have a dev environment on a remote cloud computer. I think source code is one of the most valuable assets of a company along with the data (the data more so).\n In the current environment we build the apps manually (I know it's far from being ideal, we will automate this process and I am not worried about it for now) and we deploy by simply copying the new version of top of the old one. We also do backups, recompile database stored procedures and apply scripts that massage data, change tables structures or add indexes and so on. There is some downtime allocated while the upgrades are done. Before we deploy to production we deploy to the integration and test environments.\n\n How does all this translate to an azure architecture?\n\n Here is my mental model:\n\n• Development and integration are done locally.\n• The software is packaged locally as a SAAS and uploaded to the cloud. I assume the cloud will take a manifest that includes all the resources the app needs and will allocate them when the app is created the first time in the cloud. This process will have to be tested in the cloud by us.\n• As for upgrades, the upgrades would have to be applied by the client to existing deployments. I guess another option would be to create a new instance and migrate the data from the old one. I see the upgrades as packages that can be dropped onto existing deployments and they would perform certain tasks, much like a setup software that performs an upgrade on an older version. The upgrades would have be tested as well in the cloud.\n I kind of assumed that the app can have some minor downtime. I realize that other apps might not have this luxury, but I don't want to go there now.\n Anyway, is this doable in the azure world? Should anything be done differently? And how would one charge in this model?\nThanks\n\n An update: I was thinking more about it and I guess another model would be to build a monolithic app that would allow users to sign up for the service (much like an email app) and it would allocate resources accordingly. I guess in this case the charging model is easier. The downside is that the app would have to be designed to be multi-tenant and it would have its own set of challenges. I am still fuzzy about how the whole development->test->production cycle would work in this case.",
 "label": "No"
 },
 {
 "text": "Due to cost I am currently thinking of using AWS for scalable hosting of the node.js backend of the application that would be nice to have monitoring,load balancing and all the goodies of the AWS architecture.\n\n The database system (currently OrientDB,PostgreSQL and redis) would be nice to have lots of mem and SSD disks and I have found several places with pricing at 1/4.\n\n Does anyone know of any metrics for comparing database lagging with a network request overhead or is that a non-issue in today's fast networks?",
 "label": "Yes"
 }
]

Command: With the use of the Helper-Data you will have to label the Task-Data. I want you to label the Task-Data with "Yes" if the text is related to software sustainability and with "No" in case the text fragment is unrelated". Besides labelling the post, you also need to provide a justification for the label given to the piece of text, i.e. explain why the text piece is related or not to sustainability.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "label": "<Yes/No>",
 "justification": "<Your justification>"
}

"""

both_prompt = """

Input: You will receive two kinds of text: Helper-Data and Task-Data. The Task-Data is a text fragment that you will receive as a user message. The Helper-Data is comprised of text fragments that define software sustainability.

The following two definitions represent and examples with "Yes/No" labelling in JSON format represent the following two definitions represent the Helper-Data: 
First definition: "Sustainability has multiple dimensions and here is a definition for each of them in the software intensive systems. The following paragraphs describe the dimensions used in the framework to characterize sustainability in the context of software-intensive systems: Social sustainability. Social sustainability focuses on ensuring current and future generations have the same or greater access to social resources by pursuing generational equity. For software-intensive systems, it encompasses the direct support of social communities in any domain, as well as activities or processes that indirectly create benefits for social communities; Environmental sustainability. Environmental sustainability aims to improve human welfare while protecting natural resources; for software-intensive systems, this means addressing ecological requirements, including energy efficiency and creation of ecological awareness; and Technical sustainability. Technical sustainability addresses the long-term use of software-intensive systems and their appropriate evolution in a constantly changing execution environment; and Economic sustainability. Economic sustainability focuses on preserving capital and financial value.\n An evaluation criterion can be a quality requirement. "

Second definition: " This defenition contains sustainability across its dimensions, but not specifically for software systems. The economic dimension focuses on assets, capital and added value that comprises wealth creation, prosperity, profitability, capital investment, income, etc. The environmental dimension is concerned with the long-term effects of human activities on natural systems, which includes natural ecosystems and resources, the climate, pollution and waste, etc. The social dimension covers societal communities (groups of people, organisations) and the factors that erode trust in society. The concepts analysed here encompass social equity, justice, employment, democracy, etc. The technical dimension includes the concept of the longevity of information, systems, and infrastructure and their adequate evolution within changing environmental conditions, which covers inter alia, system maintenance, obsolescence, and data integrity"

JSON formatted examples:
[
 {
 "text": "I am asking myself the exact same question at the moment.I am leaning towards the multi-instance single tenancy solution but have not taken a definitive decision yet. Let me share some of my thoughts : The main historical advantage of the multi-tenant architecture is a better use of infrastructure resources, by mutualisation (single OS, single Database, single application layer) and better occupying of said resources (when one user is away another can use the same resource). It also greatly simplifies software lifecycle : you deploy new versions to you one instance, all the customers are updated at the same time. It seems however, that recent advancements in cloud technology make the first class of advantages largely available in a multi-instance (instance-per-customer) architecture (I am thinking specifically of a platform like Jelastic here but I am sure there are others that provide the same features):\n\n • Container-based PaaS\n\n • Provisioning and auto-scaling of containers (elastic containers) \n\n So hardware and platform management is not the Software provider's concern any more. Resources are mutualised much more efficiently than before at the infrastructure and plaform levels. \n\n There will still be an overhead for multi-instance (some app and middleware will be ran N times instead of just one), but much lower than when using a separate (virtual) machine per instance. The database could be shared anyway (one schema per instance, several schemas per DB server)\nAlso :\n\n• Automation of creation of new instances is possible via PaaS API\n\n• Automation of deployment of new versions is possible via PaaS API, with zero downtime (takes some work to put in place)\n\n• Scaling is always out, never up. We don't have to worry about huge datasets at the instance level.\n\n Of course, we would need some kind of central service that manages all this automatically (e.g. creation of instance when a new user creates an account). This would also manage payment and licensing issues, interaction between instances etc. This central service might be quite complex and hard to develop, but the good thing is that we don't have to implement it upfront (now that we don't have much resource) whereas multi-tenant would need to be baked into the app from the start.\n\n Which brings me to the final advantages of developing single-tenant for a very early stage (pre-invesment) startup project :\n\n• Same (or almost same) version of the app can be deployed on-premises either as a virtual appliance or docker container or even on customer-managed machine (some companies are still reluctant towards the Cloud and it may help an early stage startup to not push out important early adopters)\n\n • Faster to get a product out with limited resources (the application layer and database schema is quite less complex), can get a 'dumb' single instance single tenant product out first (MVP) for early adopters and to show the business value of the app to potential investors, and add all the cloud automation later on \n\n• Can be seen as a selling argument for customers worried about data security : the data is better encapsulated since every customer has his own schema or even database. Much less risk of 'spillage'\n\nNB: I am obviously thinking here of a business app where customers would be businesses (each with multiple individual users) and not individuals. It would not make any sense to run a separate instance of an app for each individual user (or would it ?)",
 "label": "Yes"
 },
 {
 "text": "Node is a javascript execution environment. Node can act as a server (by executing a script that performs server functions).\nHeroku is an operating system (sitting in the cloud) that supports (among others) Node scripts. You can think of it like your desktop sitting on the internet, running a Node environment.\n When you run a server type of script in Heroku, it isn't any different than you running it on one your companies public facing servers.\n One of the nice things about Heroku over Apache, is that Heroku is designed to allow users to upload applications without an admin being required to deploy it.",
 "label": "No"
 },
 {
 "text": "We currently have a .Net MVC web app with a SQL server back-end database and two web services that perform periodically some computing tasks in the database.\n I am thinking of packaging this app for the cloud, where a client would 'instantiate' it as a whole package. This process would allocate automatically the pieces needed to run the whole thing, i.e. the site, the database and the other two services.\n I've been reading quite a bit about azure and I am having a very hard time getting my head around the whole azure development, testing and deployment process, then upgrading and applying changes to existing deployments. I am used to having control of all the pieces. First I like to have everything local on my workstation. We use git and we have our own self managed repository. I find it weird to have a dev environment on a remote cloud computer. I think source code is one of the most valuable assets of a company along with the data (the data more so).\n In the current environment we build the apps manually (I know it's far from being ideal, we will automate this process and I am not worried about it for now) and we deploy by simply copying the new version of top of the old one. We also do backups, recompile database stored procedures and apply scripts that massage data, change tables structures or add indexes and so on. There is some downtime allocated while the upgrades are done. Before we deploy to production we deploy to the integration and test environments.\n\n How does all this translate to an azure architecture?\n\n Here is my mental model:\n\n• Development and integration are done locally.\n• The software is packaged locally as a SAAS and uploaded to the cloud. I assume the cloud will take a manifest that includes all the resources the app needs and will allocate them when the app is created the first time in the cloud. This process will have to be tested in the cloud by us.\n• As for upgrades, the upgrades would have to be applied by the client to existing deployments. I guess another option would be to create a new instance and migrate the data from the old one. I see the upgrades as packages that can be dropped onto existing deployments and they would perform certain tasks, much like a setup software that performs an upgrade on an older version. The upgrades would have be tested as well in the cloud.\n I kind of assumed that the app can have some minor downtime. I realize that other apps might not have this luxury, but I don't want to go there now.\n Anyway, is this doable in the azure world? Should anything be done differently? And how would one charge in this model?\nThanks\n\n An update: I was thinking more about it and I guess another model would be to build a monolithic app that would allow users to sign up for the service (much like an email app) and it would allocate resources accordingly. I guess in this case the charging model is easier. The downside is that the app would have to be designed to be multi-tenant and it would have its own set of challenges. I am still fuzzy about how the whole development->test->production cycle would work in this case.",
 "label": "No"
 },
 {
 "text": "Due to cost I am currently thinking of using AWS for scalable hosting of the node.js backend of the application that would be nice to have monitoring,load balancing and all the goodies of the AWS architecture.\n\n The database system (currently OrientDB,PostgreSQL and redis) would be nice to have lots of mem and SSD disks and I have found several places with pricing at 1/4.\n\n Does anyone know of any metrics for comparing database lagging with a network request overhead or is that a non-issue in today's fast networks?",
 "label": "Yes"
 }
]

Command: With the use of the Helper-Data you will have to label the Task-Data. I want you to label the Task-Data with "Yes" if the text is related to software sustainability and with "No" in case the text fragment is unrelated". Besides labelling the post, you also need to provide a justification for the label given to the piece of text, i.e. explain why the text piece is related or not to sustainability.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "label": "<Yes/No>",
 "justification": "<Your justification>"
}

"""

categories_prompt = """
Input: You will receive a fragment of text which represents Task-Data, which will be a text fragment that you will receive as a user message. You will have access to a Label-List which contains multiple labels that will be used to categorize the Task-Data. The labels represent categories from the domain of Computer Science. Use at least one label to categorize the Task-Data.

Label-List = [requirements engineering, design decision, cloud services, robotics, embedded systems, computer graphics, hardware architecture, sustainability]

Command: I want you to label the Task-Data with the corresponding label from the Label-List. Besides labelling the post, you also need to provide a justification for the label you have chosen, i.e. explain why the Task-Data is a good match for the selected label.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "labels": "<labels>",
 "justification": "<Your justification>"
}

"""

categories_def_prompt = """
Input: You will receive a fragment of text which represents Task-Data, which will be a text fragment that you will receive as a user message. You will have access to a Label-List which contains multiple labels that will be used to categorize the Task-Data. Besides the labels from the list, you will receive Definitions for each entry from the list. The labels represent categories from the domain of Computer Science. Use at least one label to categorize the Task-Data.

Label-List = [requirements engineering, design decision, cloud services, robotics, embedded systems, computer graphics, hardware architecture, sustainability]

The following JSON structure contains the definitions of each label:
[
 {
 "label": "requirements engineering",
 "definition": "Requirements Engineering (RE) is the discipline that is concerned with understanding, modelling and specifying, analyzing and evolving the requirements of software systems. The Requirements Engineering Lab (RE-Lab) at Utrecht University is involved in several research directions with the common objective to help people express better requirements in order to ultimately deliver better software. Our recipe involves the use of state-of-the-art, innovative techniques from various disciplines (computer science, logics, artificial intelligence, computational linguistics, social sciences, psychology, etc.) and to apply them to solve real-world problems in the software industry."
 },
 {
 "label": "design decision",
 "definition": "An architectural design decision is therefore the outcome of a design process during the initial construction or the evolution of a software system. Architectural design decisions, among others, may be concerned with the application domain of the system, the architectural styles and patterns used in the system, COTS components and other infrastructure selections as well as other aspects needed to satisfy the system requirements."
 },
 {
 "label": "cloud services",
 "definition": "Cloud computing is a model for enabling ubiquitous, convenient, on-demand network access to a shared pool of configurable computing resources (e.g., networks, servers, storage, applications, and services) that can be rapidly provisioned and released with minimal management effort or service provider interaction. This cloud model is composed of five essential characteristics, three service models, and four deployment models.\n Essential Characteristics:\nOn-demand self-service. A consumer can unilaterally provision computing capabilities, such as server time and network storage, as needed automatically without requiring human interaction with each service provider.\n Broad network access. Capabilities are available over the network and accessed through standard mechanisms that promote use by heterogeneous thin or thick client platforms (e.g., mobile phones, tablets, laptops, and workstations).\n Resource pooling. The provider's computing resources are pooled to serve multiple consumers using a multi-tenant model, with different physical and virtual resources dynamically assigned and reassigned according to consumer demand. There is a sense of location independence in that the customer generally has no control or knowledge over the exact location of the provided resources but may be able to specify location at a higher level of abstraction (e.g., country, state, or datacenter). Examples of resources include storage, processing, memory, and network bandwidth.\n Rapid elasticity. Capabilities can be elastically provisioned and released, in some cases automatically, to scale rapidly outward and inward commensurate with demand. To the consumer, the capabilities available for provisioning often appear to be unlimited and can be appropriated in any quantity at any time.\n Measured service. Cloud systems automatically control and optimize resource use by leveraging a metering capability1 at some level of abstraction appropriate to the type of service (e.g., storage, processing, bandwidth, and active user accounts). Resource usage can be monitored, controlled, and reported, providing transparency for both the provider and consumer of the utilized service."
 },
 {
 "label": "robotics",
 "definition": "Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots.\n Within mechanical engineering, robotics is the design and construction of the physical structures of robots, while in computer science, robotics focuses on robotic automation algorithms. Other disciplines contributing to robotics include electrical, control, software, information, electronic, telecommunication, computer, mechatronic, and materials engineering.\n The goal of most robotics is to design machines that can help and assist humans. Many robots are built to do jobs that are hazardous to people, such as finding survivors in unstable ruins, and exploring space, mines and shipwrecks. Others replace people in jobs that are boring, repetitive, or unpleasant, such as cleaning, monitoring, transporting, and assembling. Today, robotics is a rapidly growing field, as technological advances continue; researching, designing, and building new robots serve various practical purposes."
 },
 {
 "label": "embedded systems",
 "definition": "An embedded system is a microprocessor-based system that is built to control a function or range of functions and is not designed to be programmed by the end user in the same way that a PC is. Yes, a user can make choices concerning functionality but cannot change the functionality of the system by adding/replac- ing software. With a PC, this is exactly what a user can do: one minute the PC is a word processor and the next it's a games machine simply by changing the software. An embedded system is designed to perform one particular task albeit with choices and different options. The last point is important because it differenti- ates itself from the world of the PC where the end user does reprogram it whenever a different software package is bought and run. However, PCs have provided an easily accessible source of hardware and software for embedded systems and it should be no surprise that they form the basis of many embedded systems. To reflect this, a very detailed design example is included at the end of this book that uses a PC in this way to build a sophisticated data logging system for a race car."
 },
 {
 "label": "computer graphics",
 "definition": "The field of computer graphics is a broad and diverse field that exists cross section between computer science and design. It is interested in the entire process of creating computer generated imagery, from creating digital three-dimensional models, to the process of texturing, rendering, and lighting those models, to the digital display of those renderings on a screen. This process starts with simple object rendering techniques to transform mathematical representations of three-dimensional objects into a two-dimensional screen image, calculating projection transformations of vertices as well as occlusion and depth of objects. Detail and realism is added to these images through simulation of materials, textures, and lighting. The most accurate and realistic techniques involve understanding the way light interacts with objects in the physical world, and simulating those interactions as closely as possible on a computer. Phenomena such as reflections, transparencies, or diffuse lighting can be modeled using a variety of different algorithms, some designed to be physically accurate, others to be computationally efficient, depending on different needs. Virtual reality imagery must be generated in a matter of milliseconds, while a detailed architectural rendering may take hours of computation time."
 },
 {
 "label": "hardware architecture",
 "definition": "Computer architecture is the organisation of the components which make up a computer system and the meaning of the operations which guide its function. It defines what is seen on the machine interface, which is targeted by programming languages and their compilers.\n There are three categories of computer architecture, and all work together to make a machine function.\n System design\nSystem design includes all hardware parts of a computer, including data processors, multiprocessors, memory controllers, and direct memory access. It also includes the graphics processing unit (GPU). This part is the physical computer system.\n \n Instruction set architecture (ISA)\n This includes the functions and capabilities of the central processing unit (CPU). It is the embedded programming language and defines what programming it can perform or process. This part is the software that makes the computer run, such as operating systems like Windows on a PC or iOS on an Apple iPhone, and includes data formats and the programmed instruction set.\n \n Microarchitecture\n Microarchitecture is also known as computer organisation and defines the data processing and storage element and how they should be implemented into the ISA. It is the hardware implementation of how an ISA is implemented in a particular processor."
 },
 {
 "label": "sustainability",
 "definition": "Sustainability has multiple dimensions and here is a definition for each of them in the software intensive systems. The following paragraphs describe the dimensions used in the framework to characterize sustainability in the context of software-intensive systems: Social sustainability. Social sustainability focuses on ensuring current and future generations have the same or greater access to social resources by pursuing generational equity. For software-intensive systems, it encompasses the direct support of social communities in any domain, as well as activities or processes that indirectly create benefits for social communities; Environmental sustainability. Environmental sustainability aims to improve human welfare while protecting natural resources; for software-intensive systems, this means addressing ecological requirements, including energy efficiency and creation of ecological awareness; and Technical sustainability. Technical sustainability addresses the long-term use of software-intensive systems and their appropriate evolution in a constantly changing execution environment; and Economic sustainability. Economic sustainability focuses on preserving capital and financial value.\n An evaluation criterion can be a quality requirement. \n \n " This definition contains sustainability across its dimensions, but not specifically for software systems. The economic dimension focuses on assets, capital and added value that comprises wealth creation, prosperity, profitability, capital investment, income, etc. The environmental dimension is concerned with the long-term effects of human activities on natural systems, which includes natural ecosystems and resources, the climate, pollution and waste, etc. The social dimension covers societal communities (groups of people, organisations) and the factors that erode trust in society. The concepts analysed here encompass social equity, justice, employment, democracy, etc. The technical dimension includes the concept of the longevity of information, systems, and infrastructure and their adequate evolution within changing environmental conditions, which covers inter alia, system maintenance, obsolescence, and data integrity"
 }
]


Command: I want you to label the Task-Data with the corresponding label from the Label-List. Besides labelling the post, you also need to provide a justification for the label you have chosen, i.e. explain why the Task-Data is a good match for the selected label.

Output: The output should be formatted in JSON format. The structure of the output needs to use the following format:
{
 "id": <Original Post ID>,
 "labels": "<labels>",
 "justification": "<Your justification>"
}

"""


df = pd.read_csv("values/check.csv")
def clean_html(text):
  return BeautifulSoup(text, "html.parser").get_text()

messages = [
  {
    "role": "user",
    "content": str(row["Id"]) + "\n \n" + clean_html(row["Body"])
  }
  for _, row in df.iterrows()
]

prompts = [zero_prompt, definition_prompt, fewshot_prompt, both_prompt, categories_prompt, categories_def_prompt]
temps = [1.0, 0.0]
efforts = ["low", "medium", "high"]

conv_counter = 0
i = 0

for t in efforts:
  for prompt in prompts:
    responses = []
    for message in messages:

      print(f"Processing message {conv_counter}.")
      print("Waiting for compeltion")
      conv_counter += 1
      completion = client.chat.completions.create(
        model = "o3-mini",
        messages=[
          {
            "role": "system", "content": prompt
          },
            message
        ],
        reasoning_effort = t,
        response_format = {"type": "json_object"}
      )
      
      print("Message completed")

      response_message = completion.choices[0].message.content
      response_json = json.loads(response_message) 
      responses.append(response_json)

    with open(f"results/o3_mini/o3_mini_temp_{t}_run_{i}_labels.json", "w") as outfile:
      json.dump(responses, outfile, indent=4)
    
    print(f"Finished Prompt {i}\n")
    i = i + 1
  i = 0

i = 0
for t in temps:
  for prompt in prompts:
    responses = []
    for message in messages:
      completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {
            "role": "system", "content": prompt
          },
            message
        ],
        temperature = t,
        response_format={"type": "json_object"}
      )
      
      response_message = completion.choices[0].message.content
      response_json = json.loads(response_message) 
      responses.append(response_json)

    with open(f"results/o3_mini/o3_mini_temp_{t}_run_{i}.json", "w") as outfile:
      json.dump(responses, outfile, indent=4)
    i = i + 1
  i = 0