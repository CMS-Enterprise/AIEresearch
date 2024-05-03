# AIE_demo_playground
An application leveraging Generative AI (Large Language Models) to assist users in finding relevant information within large documents.

## About the Project
The AIE_demo_playground project demonstrates how Generative AI (Large Language Models) can help users navigate and extract information from comprehensive documents. Through multiple approaches, the application supports both pre-loaded reference documents and user-uploaded materials, making document exploration more intuitive and efficient.

### Project Vision
This project explores how Generative AI technology can assist in navigating and extracting information from large documents. Through Large Language Models (LLMs), we investigate methods for enhancing content discovery within comprehensive documents, showcasing how AI can identify and present relevant information in accessible formats. By grounding the LLM's responses in the source document's content, we demonstrate an approach that maintains accuracy while exploring new ways to interact with extensive documentation.

### Project Mission
Four different demos were created to showcase different capabilities: (1) RAG Mistral 7B: This demo was created with a chatbot that could answer questions on the sample dataset by grounding an open-source LLM, Mistral 7B, with RAG (Retrieval-Augmented Generation). (2) Custom Document RAG: This demo is identical to the first one, but it allowed users to upload their own documents for RAG, instead of just using the sample dataset. (3) Accessibility Features: This variant leveraged AI to enable accessibility through text to speech and speech to text functionality, allowing more people to interact with the model. (4) LLM without Refinement: This demo featured an LLM operating without RAG on documents, providing a baseline for comparison.

The UI was made with Gradio and Plotly Dash, and evaluation metrics on the response and speed of the model through a LLM-assisted tool named Trulens. This was all while getting familiar with developing applications on the new KMP AWS environment, so that the team would be better prepared for future projects. Collaboratively working with KMP also led opportunities to improve instances being used on AWS.

### Agency Mission
The Centers for Medicare and Medicaid Services (CMS) has a mission to provide quality health care coverage and promote effective care for Medicare beneficiaries. 


<!-- 
### Team Mission
TODO: Good to include since this is an agency-led project -->

## Core Team

An up-to-date list of core team members can be found in [MAINTAINERS.md](MAINTAINERS.md). At this time, the project is still building the core team and defining roles and responsibilities. We are eagerly seeking individuals who would like to join the community and help us define and fill these roles.

## Documentation Index 
See the README file in the `web_app` directory to see how to run the models. 

## Repository Structure

The project is organized into a folders to encapsulate different functionalities, as illustrated below. `chat_logs` captures json files created when users interact with the chatbot. `data` includes the sample dataset used in the RAG implementation, and can be used to contain other files as well. `models` includes files to run models saved in `saved_models`. `web_app` includes the Plotly Dash website and its associated pages and components.

├── chat_logs
├── data
├── models
├── saved_models
└── web_app
    ├── assets
    └── pages
        ├── chatbot
        ├── doc_upload
        ├── qa
        └── tts


# Development and Software Delivery Lifecycle 

The following guide is for members of the project team who have access to the repository as well as code contributors. The main difference between internal and external contributions is that external contributors will need to fork the project and will not be able to merge their own pull requests. For more information on contribributing, see: [CONTRIBUTING.md](./CONTRIBUTING.md).

## Local Development

<!--- TODO - with example below:
This project is monorepo with several apps. Please see the [api](./api/README.md) and [frontend](./frontend/README.md) READMEs for information on spinning up those projects locally. Also see the project [documentation](./documentation) for more info.
-->

## Coding Style and Linters

<!-- TODO - Add the repo's linting and code style guidelines -->

Each application has its own linting and testing guidelines. Lint and code tests are run on each commit, so linters and tests should be run locally before commiting.

## Branching Model

<!--- TODO - with example below:
This project follows [trunk-based development](https://trunkbaseddevelopment.com/), which means:

* Make small changes in [short-lived feature branches](https://trunkbaseddevelopment.com/short-lived-feature-branches/) and merge to `main` frequently.
* Be open to submitting multiple small pull requests for a single ticket (i.e. reference the same ticket across multiple pull requests).
* Treat each change you merge to `main` as immediately deployable to production. Do not merge changes that depend on subsequent changes you plan to make, even if you plan to make those changes shortly.
* Ticket any unfinished or partially finished work.
* Tests should be written for changes introduced, and adhere to the text percentage threshold determined by the project.

This project uses **continuous deployment** using [Github Actions](https://github.com/features/actions) which is configured in the [./github/worfklows](.github/workflows) directory.

Pull-requests are merged to `main` and the changes are immediately deployed to the development environment. Releases are created to push changes to production.
-->

## Contributing

Thank you for considering contributing to an Open Source project of the US Government! For more information about our contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Codeowners

The contents of this repository are managed by OIT\AI Explorers. Those responsible for the code and documentation in this repository can be found in [CODEOWNERS.md](CODEOWNERS.md).

## Community

The AIE_demo_playground team is taking a community-first and open source approach to the product development of this tool. We believe government software should be made in the open and be built and licensed such that anyone can download the code, run it themselves without paying money to third parties or using proprietary software, and use it as they will.

We know that we can learn from a wide variety of communities, including those who will use or will be impacted by the tool, who are experts in technology, or who have experience with similar technologies deployed in other spaces. We are dedicated to creating forums for continuous conversation and feedback to help shape the design and development of the tool.

We also recognize capacity building as a key part of involving a diverse open source community. We are doing our best to use accessible language, provide technical and process documents, and offer support to community members with a wide variety of backgrounds and skillsets. 

### Community Guidelines

Principles and guidelines for participating in our open source community are can be found in [COMMUNITY_GUIDELINES.md](COMMUNITY_GUIDELINES.md). Please read them before joining or starting a conversation in this repo or one of the channels listed below. All community members and participants are expected to adhere to the community guidelines and code of conduct when participating in community spaces including: code repositories, communication channels and venues, and events. 

<!--
## Governance
Information about how the AIE_demo_playground community is governed may be found in [GOVERNANCE.md](GOVERNANCE.md).
-->

## Feedback

If you have ideas for how we can improve or add to our capacity building efforts and methods for welcoming people into our community, please let us know at **{contact email}**. If you would like to comment on the tool itself, please let us know by filing an **issue on our GitHub repository.**

<!--
## Glossary
Information about terminology and acronyms used in this documentation may be found in [GLOSSARY.md](GLOSSARY.md).
-->

## Policies

### Open Source Policy

We adhere to the [CMS Open Source
Policy](https://github.com/CMSGov/cms-open-source-policy). If you have any
questions, just [shoot us an email](mailto:opensource@cms.hhs.gov).

### Security and Responsible Disclosure Policy

*Submit a vulnerability:* Unfortunately, we cannot accept secure submissions via
email or via GitHub Issues. Please use our website to submit vulnerabilities at
[https://hhs.responsibledisclosure.com](https://hhs.responsibledisclosure.com).
HHS maintains an acknowledgements page to recognize your efforts on behalf of
the American public, but you are also welcome to submit anonymously.

For more information about our Security, Vulnerability, and Responsible Disclosure Policies, see [SECURITY.md](SECURITY.md).

## Public domain

This project is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/) as indicated in [LICENSE](LICENSE).

All contributions to this project will be released under the CC0 dedication. By submitting a pull request or issue, you are agreeing to comply with this waiver of copyright interest.


## Important notes
* This project used Python version 3.11.7
* This project was run on AWS using an Amazon Linux instance. OpenSSL 1.1 had to be installed. 
* An OpenAI API key is needed to run TrueLens as well as the text-to-speech and speech-to-text models. There are sections in the scripts for users to input their own key. 
* A Hugging Face token is needed to use the Mistral model. 
