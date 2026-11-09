\## What I Built

I build a RAG tool for Larkfield's Credit Union. The RAG tool is used to assist with answering questions from member services. I edited the ask.py by adding "concise yet comprehensive" to the last rule in the rules section. I did this so that the tool can return a more thorough answer. I also edited the tool to return 6 chunks instead of 4 so that it can possible return the needed information for q5 but the change did not produce the needed results. 



\## Why This Approach

Four labeled lines. If you aren't sure how your tool chunks, ask Claude Code, or open ask.py and look.

Chunking method: Structure method because it is more organized and it appears to captures what is needed the best. It captures the whole section instead of cutting things of mid-sentence or mid-paragraph.

Makes sense on its own: do your chunks make sense when read alone? Yes because it takes the section from the document verbatim therefore it reads naturally. 

Label and source: do your chunks carry a label (the document's title and section heading) and their source file? Yes my chunks carry both the label and source file. When an answer is returned, the chunk returns the title, section, and document.

Short enough: are your chunks short enough for the embedding model to read in full (about 190 words)? Start with Yes because Chroma is set to have a 256 limit for each chunk and 132 is the highest amount of words in a chunk from the Larkfield documents. 

&#x20;

\## Example Run

Paste the full contents of one of your saved runs from Questions 2 to 6, whichever you found most interesting. Don't summarize or edit it.



Rank 1 | similarity 0.526 | source fee-refund-guide.html

Larkfield Credit Union - Fee Refund Guide for Member Service Staff

Section: Refunds that need a branch manager

About this document: Version 2 (current). Owner: Member Services.



A refund needs the branch manager's approval when the fee is more than $30, when the member has already had a refund in the last 90 days, or when the member asks for several fees to be refunded at once. A branch manager may approve refunds of up to $150 in total for one member in any 90-day period.



Rank 2 | similarity 0.515 | source consumer-fee-schedule.html

Larkfield Credit Union - Consumer Fee Schedule

Section: Wire transfers

About this document: Version 8 (current). Replaces Version 7. Owner: Member Services.



\- Incoming domestic wire transfer: $10.

\- Incoming international wire transfer: $15.

\- Outgoing domestic wire transfer: $25.

\- Outgoing international wire transfer: $45. The receiving bank or an intermediary bank may deduct its own fees from the amount sent.



Rank 3 | similarity 0.494 | source fee-refund-guide.html

Larkfield Credit Union - Fee Refund Guide for Member Service Staff

Section: What a member service representative can approve

About this document: Version 2 (current). Owner: Member Services.



A member service representative may approve a refund of a single fee of $30 or less without anyone else's approval. A representative may approve only one refund for the same member in any 90-day period.



Rank 4 | similarity 0.475 | source fee-refund-guide.html

Larkfield Credit Union - Fee Refund Guide for Member Service Staff

Section: Refunds that need the Member Services Director

About this document: Version 2 (current). Owner: Member Services.



Refunds of more than $150 in total for one member in any 90-day period need the Member Services Director's approval. So does any refund for a business account.



Rank 5 | similarity 0.456 | source fee-refund-guide.html

Larkfield Credit Union - Fee Refund Guide for Member Service Staff

Section: Purpose of this guide

About this document: Version 2 (current). Owner: Member Services.



Members sometimes ask Larkfield to refund a fee. This guide explains who on staff can approve a fee refund and how to record it, so that refunds are fair and consistent across branches.



Rank 6 | similarity 0.445 | source fee-refund-guide.html

Larkfield Credit Union - Fee Refund Guide for Member Service Staff

Section: Recording a refund

About this document: Version 2 (current). Owner: Member Services.



Every refund must be recorded in the member's account notes with the fee, the amount refunded, the reason, and the name of the person who approved it. Refunds that are not recorded this way may be reversed during the monthly audit.



ANSWER

No. The outgoing international wire transfer fee is $45 \[consumer-fee-schedule.html], and a member service representative can only approve refunds of $30 or less without additional approval \[fee-refund-guide.html]. Since $45 exceeds $30, it requires branch manager approval \[fee-refund-guide.html]. It would need the Member Services Director's approval instead if it's a business account or if the member's total refunds in the last 90 days would exceed $150 \[fee-refund-guide.html].



\## Reflection

Four labeled lines, about Questions 1 and 6.

What was wrong: There were two documents labeled as consumer-fee-schedule which resulted in an error. There is version 7 and a version 8. Once I discovered the error. 

How I fixed it: I removed version 7 and rebuilt the database therefore the tool will only pull from the most updated document which in this case is version 8.

Prevention: Including the version label in the document name would help Larkfield because the tool will be able to pull from the most updated version if instructed to do so. I think this will also be helpful if the company needs to refer to older versions, simply stating the specific allows the assistant to know exactly where to retrieve from.

Question 6: Declined. The assistant stated that there were no documents that were able to answer the question. It stated that the top chunk was "Dividend rates" for share certificates (0.271) but instead of guessing and giving a percentage from that section to sound confident, it notified us that there were no documents that can provide the answer like instructed. 





