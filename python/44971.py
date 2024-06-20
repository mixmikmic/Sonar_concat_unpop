from IPython.display import Image


Image(filename='../visualizations/visualization11.png')


# <h1> Theme 11: Make No Changes </h1>
# 
# <h2> Transition 1 (Top) </h2> web.entry -> web.exit
# <h4> What's going on</h4> the user goes to the page and then leaves. 
# <br><br>
# 
# <h2> Transition 2 </h2> web.exit -> agent.entry
# 
# <h4> What's going on </h4> The user exits the site. An agent enters the user's account.
# <br><br>
# 
# <h2> Transition 3 </h2> agent.view.statments -> agent.view.history
# 
# <h4> What's going on </h4> The agent views the user's statements and history.
# <br><br>
# 
# <h2> Transition 4 </h2> journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user starts by going to the website.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> All of these actions involve no change being made. All of these events could be put in the order journey.entry-> web.entry -> web.exit -> agent.entry -> agent.view.statements -> agent.view.history
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization1.png')


# <h1> Theme 1: 
# 
# <h2> Transition 1 (Top) </h2> webevent.system.retrieve.default.email.address -> webstc.view.print.download.year.end.statement
# 
# <h4> What's going on</h4> The user is selecting their default email address and then viewing their annual statement.
# <br><br>
# 
# <h2> Transition 2 </h2> webstc.promotional.offer.eligibility -> webstc.select.a.promotional.offer
# 
# <h4> What's going on </h4> The user recieves a promotional offer and then selects it.
# <br><br>
# 
# <h2> Transition 3 </h2> webevent.open.statement.success -> webevent.login
# 
# <h4> What's going on </h4> The user looks at their statement and then logs in. This is not the expected order of events.
# <br><br>
# 
# <h2> What's Interesting</h2> Selecting a promotion is in the same cluster as users viewing their annual statement. One hypothesis is that this is occurring because people see their statements and want to get better deals, advertised by the promotional offer. 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# <li>Skew promotional ads toward people who have looked at their statements
# <li>Include promotional ads on user statement pages
# <li>Include notes on how the statement could change given a certain promotion
# 




from IPython.display import Image


Image(filename='../visualizations/visualization8.png')


# <h1> Theme 8: 
# 
# <h2> Transition 1 (Top) </h2> webevent.update.bank.profile.success -> webevent.view.make.a.payment
# <h4> What's going on</h4> The user updates their bank profile and then goes to the makes a payment page.
# <br><br>
# 
# <h2> Transition 2 </h2> web.entry -> mobile.entry
# 
# <h4> What's going on </h4> The user interacts with the web platform and then interacts with the mobile platform.
# <br><br>
# 
# <h2> Transition 3 </h2> webevent.make.a.payment.success -> web.entry
# 
# <h4> What's going on </h4> The user goes from making a payment to the home page. 
# <br><br>
# 
# <h2> Transition 4 </h2> webevent.view.transactions.and.details.success -> mobile.entry
# 
# <h4> What's going on </h4> The user views their transactions online and then logs in on the mobile platform.
# <br><br>
# 
# <h2> Transition 5 </h2> journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user begins their journey by logging onto the web
# <br><br>
# 
# <h2> Transition 6 </h2> webstc.chat.reactive -> webstc.view.print.download.monthly statement
# 
# <h4> What's going on </h4> The user uses the chat bot and then downloads their monthly statement.
# <br><br>
# 
# <h2> What's Interesting</h2> These don't seem to be highly related themes. The prevailing theme is entering a mobile or web platform. These might be interactions toward the beginning of a journey.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization5.png')


# <h1> Theme 5: Web Logins and Deleting Payments
# 
# <h2> Transition 1 (Top) </h2> mobile.exit -> web.entry
# 
# <h4> What's going on</h4> Users are interacting with a mobile platform. They leave the mobile platform to get onto the web platform.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.payment.activity -> webevent.delete.payment.success
# 
# <h4> What's going on </h4> The user sees their payment activity and then deletes a payment from the history. 
# <br><br>
# 
# <h2> Transition 3 </h2> webevent.delete.payment.success -> webevent.view.payment.activity
# 
# <h4> What's going on </h4> The user goes from deleting a payment to viewing the payment history to verify that the payment was deleted. 
# <br><br>
# 
# <h2> Transition 4 </h2> web.exit -> tsys.update.mailing.address
# 
# <h4> What's going on </h4> 
# <br><br>
# The user logs out and then their mailing address is updated.
# 
# 
# <h2> What's Interesting</h2> There are very strong probabilities of the top 2 words words associated with this cluster; however, it is unclear how these words are related: logging off of mobile and onto the web  and then deleting payments. 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# Not quite sure yet.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization16.png')


# <h1> Theme 16: Just Show Me the Summary</h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.system.retrieve.default.email.address -> webstc.view/print/download.monthly.statement
# <h4> What's going on</h4> The website retrieves the user's email and takes the user to their monthly statement.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.login -> webevent.view.account.summary.success
# 
# <h4> What's going on </h4> The user logs on online.
# <br><br>
# 
# <h2> Transition 3 </h2>  web.entry -> webevent.login
# 
# <h4> What's going on </h4> The user logs in online.
# <br><br>
# 
# <h2> Transition 4 </h2> web.exit -> journey.exit
# 
# <h4> What's going on </h4> The user exits online, leaving the journey.
# <br><br>
# 
# <h2> Transition 5 </h2> webstc.view/print/download.monthly.statement -> webevent.open.statement.success
# 
# <h4> What's going on </h4> The user opens their statement.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is an online cluster. The majority of it is users logging in to view their statements. The top 5 transitions could be brought together to build a journey: webevent.login -> view.account.summary -> retrieve.default.email -> monthly.statement -> open.statement.success -> web.exit -> journey.exit
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# Since there are a lot of people logging in just to see their monthly statement, companies should optimize their websites for accesing the statment. For example, they could include a big link to the statement near the center of the account summary. 
# 
# Looking at my Wells Fargo account, this is exactly how their new website design works. They actually just redesigned their website making the links to the statements more central and larger.
# 




from IPython.display import Image


Image(filename ='../visualizations/visualization4.png')


# Throwaway Theme. 
# 




from IPython.display import Image


Image(filename='../visualizations/visualization26.png')


# Throwaway III
# 




from IPython.display import Image


Image(filename='../visualizations/visualization23.png')


# <h1> Theme 23: Rewards </h1>
# 
# <h2> Transition 1 (Top) </h2> reward -> web.entry
# <h4> What's going on</h4> The user recieves an award and the logs in online.
# <br><br>
# 
# <h2> Transition 2 </h2> reward -> reward
# 
# <h4> What's going on </h4> The user recieves a reward.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a reward cluster. There is almost a .5 probability of this cluster if we have "reward -> web entry". 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization13.png')


# <h1> Theme 13: We Need Your Default Email for That</h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.view.statements.page -> webstc.view/print/download.monthly.statement
# <h4> What's going on</h4> the user views their statement.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.login -> webevent.view.account.summary.success
# 
# <h4> What's going on </h4> The user logs in and views their account summary.
# <br><br>
# 
# <h2> Transition 3 </h2> webevent.view.transactions.and.details.success -> webevent.system.retrieve.default.email.adress
# 
# <h4> What's going on </h4> The user goes from their transaction history to retrieving their default email.
# <br><br>
# 
# <h2> Transition 4 </h2> web.entry -> webevent.login
# 
# <h4> What's going on </h4> The user goes to the home page and logs in.
# <br><br>
# 
# <h2> Transition 5 </h2> webevent.view.account.summary.success -> webevent.system.retrieve.default.email.address
# 
# <h4> What's going on </h4> The user goes from their account summary to retrieving their default email.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> Why is the system retrieving the user's default email address? 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization28.png')


# <h1> Theme 28: Email Settings</h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.system.retrieve.default.email.address -> webevent.view.statement.and.document.delivery.preferences
# <h4> What's going on</h4> The user goes to their delivery preferences.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.statement.and.document.delivery.preferences -> webevent.system.retrieve.default.email.address
# 
# <h4> What's going on </h4> The user views their email preferences.
# <br><br>
# 
# <h2> Transition 3 </h2> web.exit -> journey.exit
# 
# <h4> What's going on </h4> The exits the website, ending their journey.
# <br><br>
# 
# 
# 
# <h2> What's Interesting</h2> This is the cluster of users going to their email preferences. 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization17.png')


# <h1> Theme 17: </h1>
# 
# <h2> Transition 1 (Top) </h2> tsys.returned.mail -> tsys.update.mailing.address
# <h4> What's going on</h4> The system automatically updating a user's address due to returned mail.
# <br><br>
# 
# <h2> Transition 2 </h2> tsys.update.mailing.address -> tsys.update.mailing.address
# 
# <h4> What's going on </h4> The system updating the user's address.
# <br><br>
# 
# <h2> Transition 3 </h2>  journey.entry -> tsys.card.request
# 
# <h4> What's going on </h4> A journey begins with a card charge request.
# <br><br>
# 
# <h2> Transition 4 </h2> webevent.login -> webevent.view.account.summary.success
# 
# <h4> What's going on </h4> The user logs in.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a tsys cluster. The tsys automatically updates the user's address when mail is returned.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# It would be useful if the tsys sent an alert to the mobile/web serv
# 

from IPython.display import Image


Image(filename='../visualizations/visualization22.png')


# 
# 
# 
# 
# <h1> Theme 22: Throwaway 2 </h1>
# 
# <h2> Transition 1 (Top) </h2> web.entry -> webevent.login
# <h4> What's going on</h4> The user logs in.
# <br><br>
# 
# <h2> Transition 2 </h2> web.exit -> journey.exit
# 
# <h4> What's going on </h4> The user logs out, ending their journey.
# <br><br>
# 
# <h2> Transition 3 </h2>  journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user begins their journey by going to the site.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> N/A  
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization24.png')


# <h1> Theme 24: Declined and Increased Credit </h1>
# 
# <h2> Transition 1 (Top) </h2> webstc.request.clip.-declined -> web.exit
# <h4> What's going on</h4> The user's request is declined and the exit the site.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.transactions.and.details.success -> webstc.view.credit.line.increase.page
# 
# <h4> What's going on </h4> The user goes to their credit line increase page from their transactions page.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.make.a.payment.success -> webstc.view.rewards.summary
# 
# <h4> What's going on </h4> The user makes a payment and views their rewards.
# <br><br>
# 
# <h2> Transition 2 </h2> journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user begins their journey by loggin in online.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> Need more information.  
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization27.png')


# 
# <h1> Theme 27: Mobile Mania </h1>
# <h2> Transition 1 (Top) </h2> mobile.entry -> mobileevent.login.with.password.success
# <h4> What's going on</h4> The user logs in on mobile with their password.
# <br><br>
# 
# <h2> Transition 2 </h2> mobileevent.view.transactions.and.details -> mobileevent.view.make.a.payment
# 
# <h4> What's going on </h4> The user makes a payment via mobile.
# <br><br>
# 
# <h2> Transition 3 </h2> reward -> mobile.entry
# 
# <h4> What's going on </h4> The user logs in online after recieving an award.
# <br><br>
# 
# <h2> Transition 4 </h2> mobileevent.view.account.summary.success -> mobileevent.view.transactions.and.details
# 
# <h4> What's going on </h4> The user views their transactions via mobile.
# <br><br>
# 
# <h2> Transition 5 </h2> mobileevents.view.make.a.payment -> mobile.exit
# 
# <h4> What's going on </h4> The user views their make a payment page and then exits the app.
# <br><br>
# 
# <h2> Transition 6 </h2> mobile.exit -> journey.exit
# 
# <h4> What's going on </h4> Exiting the app ends the user's journey.
# <br><br>
# 
# <h2> Transition 7 </h2> mobileevent.view.account.summary.success -> mobileevent.view.make.a.payment
# 
# <h4> What's going on </h4> The user makes a mobile payment after viewing their account summary.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is the cluster of users who use almost exclusive mobile banking. There is a clear path here of going from their account summary to make a payment and then logging out. 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization2.png')


# <h1> Theme 1: Agent Actions
# 
# <h2> Transition 1 (Top) </h2> agent.view.history -> agent.view.history
# 
# <h4> What's going on</h4> The agent is viewing their history. Perhaps they step away from the computer and then go back to the history again.
# <br><br>
# 
# <h2> Transition 2 </h2> agent.view.history -> agent.exit
# 
# <h4> What's going on </h4> The agent views their history and then exits.
# <br><br>
# 
# <h2> Transition 3 </h2> agent.view.statements -> agent.view.payments.history
# 
# <h4> What's going on </h4> The agent views their statements and then payment history. The agent is going directly from amount due to previous payments.
# <br><br>
# 
# <h2> Transition 4 </h2> agent.exit -> journey.exit
# 
# <h4> What's going on </h4> The agent exits, ending the journey. This signals that this journey was driven by the agent.
# <br><br>
# 
# <h2> Transition 5 </h2> agent.view.payment.history -> agent.exit
# 
# <h4> What's going on </h4> The agent is viewing the payment history and then exiting. 
# <br><br>
# 
# <h2> Transition 6 </h2> agent.pay.by.phone.success -> agent.view.payment.history
# 
# <h4> What's going on </h4> The agent is paying by phone and then verifying the payment by looking at the payment history.
# <br><br>
# 
# <h2> What's Interesting</h2> This cluster pertains almost exclusively to agents. Even the lower probability words,  (P(word) less than .05 ), are related to agents. That being said, most words in this cluster also pertain to history and payment history.
# <br><br>
# 
# <h2> What we can do with this data </h2>
# If we want to optimize agent efficiency then we should make sure that we have links to history and payment history from as many pages as possible. Is this currently the case?
# 




from IPython.display import Image


Image(filename='../visualizations/visualization19.png')


# <h1> Theme 18: Help Please! </h1>
# 
# 
# 
# 
# 
# <h2> Transition 1 (Top) </h2> web.exit -> agent.exit
# <h4> What's going on</h4> The user exits the webpage and then the agent exits.
# <br><br>
# 
# <h2> Transition 2 </h2> webstc.chat-reactive -> agent.entry
# 
# <h4> What's going on </h4> The agent enters to perform a web chat with the user.
# <br><br>
# 
# <h2> Transition 3 </h2>  webevent.view.transactions.and.details.success -> ivr.entry
# 
# <h4> What's going on </h4> The user views their transactions and details and then calls the IVR.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> These are all interactions where the user is interacting with the website and then reaches out. Unfortunately the probabilities of these words are low and therefore the predictive probability of anything outside of the top 3 transitions is very low.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization7.png')


# <h1> Theme 7: 
# 
# <h2> Transition 1 (Top) </h2> ivr.disp.completed.call -> ivr.exit
# <h4> What's going on</h4> The user completes their call and then exit.
# <br><br>
# 
# <h2> Transition 2 </h2> web.exit -> ivr.entry
# 
# <h4> What's going on </h4> The user logs off of the web platform and then calls the company. 
# <br><br>
# 
# <h2> Transition 3 </h2> agent.view.authorization -> agent.view.authorization
# 
# <h4> What's going on </h4> The agent looks at the user's authorization. 
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This seems to be the cluster of things going right with IVR, given that the ivr.exit follows a completed call. 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# These are examples of where changes don't need to be made.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization29.png')


# Throwaway IV
# 




from IPython.display import Image


Image(filename='../visualizations/visualization6.png')


# <h1> Theme 5: Web Logins and Deleting Payments
# 
# <h2> Transition 1 (Top) </h2> mobileevent.login.with.password.success -> mobile.exit
# 
# <h4> What's going on</h4> Users are logging in, looking at the home screen, and then loggin back out.
# <br><br>
# 
# <h2> Transition 2 </h2> mobile.exit -> journey.exit
# 
# <h4> What's going on </h4> The user logs off the mobile site and that is the end of the journey. 
# <br><br>
# 
# <h2> Transition 3 </h2> web.exit -> mobile.entry
# 
# <h4> What's going on </h4> The user exits the web platform and then makes an entry on the mobile platform.
# <br><br>
# 
# <h2> Transition 4 </h2> mblstc.view.services -> mobile.exit
# 
# <h4> What's going on </h4> 
# <br><br>
# The user is viewing possible services on the mobile platform and then exiting.
# 
# 
# <h2> What's Interesting</h2> There is a strong association with mobile interaction and exiting. People tend to do things faster on their phones, but it looks like a lot of people are accessing the app and leaving without taking any substantial action. This points to a cluster of users who are downloading the mobile app but don't know how to use it.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# Do these mobile apps have interactive tutorials for beginners?

from IPython.display import Image


Image(filename='../visualizations/visualization15.png')


# <h1> Theme 15: Mobile Users</h1>
# 
# <h2> Transition 1 (Top) </h2> mobileevent.login.with.sureswipe.success -> mobileevent.view.summary.success
# <h4> What's going on</h4> the user logs in using sure swipe.
# <br><br>
# 
# <h2> Transition 2 </h2> mobile.exit -> mobile.entry
# 
# <h4> What's going on </h4> The user logs off and then logs back on.
# <br><br>
# 
# <h2> Transition 3 </h2>  mobileevent.view.account.summary.success -> mobile.exit
# 
# <h4> What's going on </h4> The user exits from their account summary page
# <br><br>
# 
# <h2> Transition 4 </h2> mobileevent.view.account.summary.success -> mobile.view.transactions.and.details
# 
# <h4> What's going on </h4> The user goes from their account summary to their transactions and details page.
# <br><br>
# 
# <h2> Transition 5 </h2> mobileevent.view.transactions.and.details -> mobileevent.make.a.payment
# 
# <h4> What's going on </h4> The user goes from their transaction page to make a payment
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a mobile cluster. Here we finally have users interacting with the mobile platform. What's interesting is that using the swipe login is a sign of proficiency with the mobile app.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# If we have a user interacting with this cluster, we can categorize this user as a mobile user. They know how to interact with the mobile platform.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization21.png')


# <h1> Theme 21: Statements, Statements, Statements</h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.open.statement.success -> webstc.view/print/download.monthly.statement
# <h4> What's going on</h4> The user views their statement.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.statements.page -> webstc.view/print/download.monthly.statement
# 
# <h4> What's going on </h4> The user views their statement.
# <br><br>
# 
# <h2> Transition 3 </h2>  webstc.view/print/download.monthly.statement -> webevent.view.make.payment
# 
# <h4> What's going on </h4> The user views their statement and goes to make a payment.
# <br><br>
# 
# <h2> Transition 4 </h2>  webstc.view/print.download.monthly.statement -> webevent.view.payment.activity
# 
# <h4> What's going on </h4> The user views their statement and then views their payment activity.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> These are all centered around viewing the monthly statement.  
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization20.png')


# <h1> Theme 20: The Shallow Web</h1>
# 
# 
# 
# 
# <h2> Transition 1 (Top) </h2> web.entry -> webevent.login
# <h4> What's going on</h4> The user logs in.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.account.summary.success -> webevent.view.transactions.and.details.success
# 
# <h4> What's going on </h4> The user goes to view their transactions and details.
# <br><br>
# 
# <h2> Transition 3 </h2>  web.exit -> journey.exit
# 
# <h4> What's going on </h4> The user logs out, ending their journey.
# <br><br>
# 
# <h2> Transition 4 </h2>  journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user begins their journey by loggin onto the website.
# <br><br>
# 
# <h2> Transition 5 </h2>  webevent.view.transactions.and.details.success -> web.entry
# 
# <h4> What's going on </h4> The user views their transactions and details and then goes to the home page.
# <br><br>
# 
# <h2> Transition 6 </h2>  webevent.view.account.summary.success -> web.exit
# 
# <h4> What's going on </h4> The user views their account summary and exits
# <br><br>
# 
# <h2> What's Interesting</h2> These are all very shallow web interactions.  
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization14.png')


# <h1> Theme 14: Mobile Disengagement Part II</h1>
# 
# <h2> Transition 1 (Top) </h2> web.exit -> mobile.entry
# <h4> What's going on</h4> the user exits the online platform and goes onto the mobile platform.
# <br><br>
# 
# <h2> Transition 2 </h2> mobileevent.login.with.password.success -> mobileevent.view.account.summary.success
# 
# <h4> What's going on </h4> The user logs in and views their account summary on the mobile platform.
# <br><br>
# 
# <h2> Transition 3 </h2>  mobile.exit -> journey.exit
# 
# <h4> What's going on </h4> The user logs off mobile, and ends their journey.
# <br><br>
# 
# <h2> Transition 4 </h2> mobile.exit -> mobile.entry
# 
# <h4> What's going on </h4> The user is leaving the site but then going back (mobile).
# <br><br>
# 
# <h2> Transition 5 </h2> mobile.event.login.with.password.success -> mobile.event.answer.security.question
# 
# <h4> What's going on </h4> The user is logging in and answering the security question on their phone.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a mobile cluster. Most actions, however, are simply logging on and logging off.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# This is the same mobile engagement problem seen in theme 6. Therefore, recommendations are the same.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization18.png')


# <h1> Theme 18: Make a Payment </h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.login -> webevent.view.account.summary.success
# <h4> What's going on</h4> The user logs in.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.payment.activity -> webevent.make.a.payment.success
# 
# <h4> What's going on </h4> The user looks at their payment activities and then makes a payment.
# <br><br>
# 
# <h2> Transition 3 </h2>  webevent.view.account.summary.success -> webevent.view.make.a.payment
# 
# <h4> What's going on </h4> The user goes from the account summary to the make a payment page.
# <br><br>
# 
# <h2> Transition 4 </h2> web.entry -> webevent.login
# 
# <h4> What's going on </h4> The user logs in.
# <br><br>
# 
# <h2> Transition 5 </h2> web.exit -> journey.exit
# 
# <h4> What's going on </h4> The user logs out, ending the journey.
# <br><br>
# 
# <h2> Transition 6 </h2> webevent.view.payment.activity -> journey.view.make.a.payment
# 
# <h4> What's going on </h4> The user views their payments and goes to make a payment
# <br><br>
# 
# <h2> Transition 7 </h2> journey.entry -> web.entry
# 
# <h4> What's going on </h4> The user starts the journey by going to the web page.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a web cluster. Specifically this is web usage to make payments.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization3.png')


# <h1> Theme 1: The Horrendous Mess that is Interactive Voice Recordings
# 
# <h2> Transition 1 (Top) </h2> ivr.disp.agent.transfer -> ivr.exit
# 
# <h4> What's going on</h4> User is on the phone with an agent. The agent then transfers the user and the user exits. Most likely, the user goes onto hold and then exits while on hold.
# <br><br>
# 
# <h2> Transition 2 </h2> agent.1 -> ivr.disp.agent.transfer
# 
# <h4> What's going on </h4> An agent is assigned an item and then transfers the call. 
# <br><br>
# 
# <h2> Transition 3 </h2> ivr.entry -> ivr.proactive.balance
# 
# <h4> What's going on </h4> The user selects the option taking them to their balance. 
# <br><br>
# 
# <h2> Transition 4 </h2> ivr.proactive.balance -> agent.entry
# 
# <h4> What's going on </h4> The user selects the option taking them to their balance and then an agent submits data on the user's behalf.
# <br><br>
# 
# <h2> Transition 5 </h2> agent.exit -> journey.exit
# 
# <h4> What's going on </h4> The agent is viewing the payment history and then exiting. 
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is clearly an IVR (interactive voice recording) cluster. The top words associated with this cluster are: 
# <li>users getting transferred and then exiting
# <li>Agents transferring the user
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# Clearly call transfers are a real problem. Everyone has experienced being transferred around endlessly while attempting to complete a simple task with the company. What about an experiment where agents weren't allowed to transfer calls? In the least, we should be working to make transferring calls more difficult.
# 




from IPython.display import Image


Image(filename='../visualizations/visualization25.png')


# <h1> Theme 25: Increased Credit For You  </h1>
# 
# <h2> Transition 1 (Top) </h2> webstc.view.credit.line.increase.page -> webevent.request.credit.limit.increase
# <h4> What's going on</h4> The user requests a credit increase.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.request.credit.limit.increase -> webstc.request.clip-approved.-accepted
# 
# <h4> What's going on </h4> The user's credit increase request is approved.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.account.summary.success -> webstc.request.clip.-approved.-accepted
# 
# <h4> What's going on </h4> The user's credit increase request is approved.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.transactions.and.details.success -> webstc.view/redeem.rewards
# 
# <h4> What's going on </h4> The user views their rewards.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.request.credit.limit.increase -> marketing
# 
# <h4> What's going on </h4> The user requests a credit limit increase and goes to marketing.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> This is a cluster of people who recieve increased credit. Interesting people who are requesting a credit increase are associated with checking their rewards (regardless of whether they recieve the credit increase or not). 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization10.png')


# <h1> Theme 10: Browsing and Chargebacks </h1>
# 
# <h2> Transition 1 (Top) </h2> webstc.browse.us.cards -> webevent.login
# <h4> What's going on</h4> The user is browing credit cards and then logs in 
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.use.search.bar.success -> webstc.search
# 
# <h4> What's going on </h4> The user uses the search bar and is directed to a search.
# <br><br>
# 
# <h2> Transition 3 </h2> webstc.view.rewards.summary -> webstc.manage.rewards
# 
# <h4> What's going on </h4> The user views their rewards and then goes to manage rewards.
# <br><br>
# 
# <h2> Transition 4 </h2> tsys.chargeback -> tsys.financial.adjustment
# 
# <h4> What's going on </h4> A chargeback causes a financial adjustment in the tsys system.
# <br><br>
# 
# <h2> Transition 5 </h2> tsys.chargeback -> tsys.financial.adjustment
# 
# <h4> What's going on </h4> A financial adjustment in the tsys system causes a chargeback.
# <br><br>
# 
# <h2> What's Interesting</h2> Why are chargebacks clustered with users browsing options on their accounts?
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A




from IPython.display import Image


Image(filename='../visualizations/visualization12.png')


# <h1> Theme 12:  </h1>
# 
# <h2> Transition 1 (Top) </h2> webevent.modify.alerts.success -> webevent.view.alerts.page
# <h4> What's going on</h4> the user modifies their alerts and then goes back to the alerts page.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.alerts.page -> webstc.view/edit/add.alert.contact.points
# 
# <h4> What's going on </h4> The user goes from their alerts page to editing their alerts.
# <br><br>
# 
# <h2> Transition 3 </h2> webevent.view.alerts.page -> webevent.modify.alerts.success
# 
# <h4> What's going on </h4> The user goes from viewing their alerts to modifying their alerts.
# <br><br>
# 
# <h2> Transition 4 </h2> webevent.view.account.summary.success -> webevent.view.alerts.page
# 
# <h4> What's going on </h4> The user goes from their account summary to the alerts page.
# <br><br>
# 
# 
# <h2> What's Interesting</h2> These are all related to user alerts. They are very neatly packed around the user changing their alerts. 
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# N/A
# 




from IPython.display import Image


Image(filename='../visualizations/visualization9.png')


# <h1> Theme 9: Timing Out and Logging Back In
# 
# <h2> Transition 1 (Top) </h2> webevent.view.account.summary.success -> webevent.login
# <h4> What's going on</h4> The user views their account summary and then logs in. They may have timed out and have to log in again.
# <br><br>
# 
# <h2> Transition 2 </h2> webevent.view.account.summary.success -> webevent.view.transactions.and.details.success
# 
# <h4> What's going on </h4> The user goes from viewing their account summary to viewing their transactions and details.
# <br><br>
# 
# <h2> Transition 3 </h2> web.entry -> webevent.login
# 
# <h4> What's going on </h4> The user goes to the web page and then logs in.
# <br><br>
# 
# <h2> Transition 4 </h2> webevent.view.transactions.and.details.success -> webevent.login
# 
# <h4> What's going on </h4> The user views their transactions online and then logs in on the web.
# <br><br>
# 
# <h2> What's Interesting</h2> You have to be logged in to view your transactions page or account summary. This means that these interactions are users staying on a page until they are auto logged off. They then have to log on again.
# 
# <br><br>
# 
# <h2> What we can do with this data </h2>
# This cluster doesn't seem to present a clear problem. Logging back into your account because you remain on a page for too long is good for security. 
# 




