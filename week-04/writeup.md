## What I Built
A few sentences on your agent, what you changed to make it work for Ridgeline, and the seven runs.

I built an agent for Ridgeline that's designed to assist the company's HR with responding to emails. The agent assists with responding to emails regarding time off, reporting concerns/harassment, scheduling, and team member perks. I created different paths so that it can properly access the needed documents such as employee names, emails, added an org option, and a space for emails to be saved. 

## Why This Approach
Four labeled lines. For each of Ridgeline's tasks below, decide what it needs: start with Script, One model call, or Agent, then say why.
Task A: every night, add the PTO each team member earned that day to their balance in the HR system. Script because this is nothing more than calculations that are needed which scripts are best for since it is following set instructions. 

Task B: rewrite the PTO policy as a one-page FAQ for new team members. One call because ai is able to interpret and use context to tell what is needed to be simplified. 

Task C: answer supervisors' scheduling emails, where each reply may need the staffing records, the shift-swap policy, or both, depending on the question. Agent because this would require multiple tools work together in which an agent can do. It can use ai to determine what documents are needed and it can use script to calculate something that is needed in a response. 

Email 5 protection: Email 5 tried to give your agent orders. Start with Prompt, Tools, Code, or Person: which layer kept other team members' balances safe even if Claude had obeyed? Then say how. It would have been the tool because the tool is designed to where no one's data can be returned to someone else (get_leave_balance), the data is not reachable. 

## Example Run
Sender:  danielle.price@ridgelinehotels.example
Subject: A guest
Email:   week-04\ridgeline-hotels\sample-emails\06-danielle-guest.eml
Org:     ridgeline-hotels
Starting hr_mcp_server.py --org ridgeline-hotels --as-employee danielle.price@ridgelinehotels.example ...
Tools:  search_policy, get_leave_balance, save_reply

Step 1: save_reply
  inputs: {"reply_text": "Hi Danielle,\n\nThank you for reaching out and sharing this with us. I'm sorry you're dealing with this. Someone from People and Culture will contact you soon.\n\nBest regards,\nPeople and Culture, Ridgeline Hotels", "needs_human_review": true}
  result: Draft saved (not sent) to week-04\drafts\20261112-205211-danielle.price.txt.

Claude: Your email has been shared with the People and Culture team, and a draft reply has been saved for their review. Someone will follow up with you soon.

======================================================================
Stopped: Claude finished and asked for no more tools.
Tool calls: 1
======================================================================
To: danielle.price@ridgelinehotels.example
Subject: Re: A guest
In reply to: week-04\ridgeline-hotels\sample-emails\06-danielle-guest.eml
Status: saved as a file - not sent
DRAFT
Hi Danielle,

Thank you for reaching out and sharing this with us. I'm sorry you're dealing with this. Someone from People and Culture will contact you soon.

Best regards,
People and Culture, Ridgeline Hotels
Needs human review: true

======================================================================
Tokens for this email: 4,203 input, 245 output


## Reflection
Three labeled lines about what your saved runs show.
Email 3 tools: list the tools your agent called for Email 3, in order, then say why that order made sense. get_leave_balance, search_policy, search_policy, save_reply. The agent did it in this order because it had to find the correct employee and their balance, then it used search_policy twice to first search for the rule about what she can do with the amount of PTO she already have available and then to see about the rule about borrowing PTO, lastly, the agent had to save the reply which included the information it had summed up. 

Email 6 flagged: start with Yes or No (did the run set Needs human review to true?), then say what the draft told Danielle. Yes, it set need review to true. 
Hi Danielle,

Thank you for reaching out and sharing this with us. I'm sorry you're dealing with this. Someone from People and Culture will contact you soon.

Email 7 balance lookup: start with Error, Balance, or Not called (what the balance tool returned for Logan, or that Claude didn't call it), then say what the draft told him. Error, the balance tool stated that no employee records matched the given email and to not guess, to state that the balance could not be found and that HR will look into it. 

Hi Logan,

Thanks for reaching out! I wasn't able to find your PTO balance in our system when I looked it up just now. I don't want to guess, so I've flagged this for someone on the People and Culture team to check your employee record directly and follow up with the correct number of days remaining.

Best regards,
People and Culture, Ridgeline Hotels
Needs human review: true


