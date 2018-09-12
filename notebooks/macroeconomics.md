---
layout: default_sidebar
title: GHBCode's Site
description: Notes on Comp Sci, Math, Data Analysis, Python and Misc
---

## Macroeconomics

Notes taken from the book "A Concise Guide to Macroeconomics" by David A. Moss. 


### Comparative advantage
* **TFP total factor productivity:** increase output via increase in labor, capital or efficiency
  * **Interesting excercise** where you have country A and country B and items X and Y. 
    * Say country A is more efficient at producing both X and Y, when compared to country B. 
    * Country A is most efficient at producing item X.
    * It is more beneficial (for all) for country A to solely produce item X and country B to produce Y.
      * When production is managed this way between both countries, they produce as much as possible and therefore benefit from the largest output possible. 
* Supply siders vs government led as a source of increase in TFT (lowering taxes vs gov't stimulation)
* **Market economy:** highly decentralized where each household decides on saving and investing


### Money
* **Price of money:** interest rate, FX rate, aggregate price level (overall level of prices in the economy)
* **Nominal vs real GDP:** Real GDP increases only when Q(quantity) of output increases
  * Cow example: output not money is what matters. 
    * Inflation can errode your purchasing power, i.e. 
      * lend 10 cows and one year later receive 11 vs.
      * lend 10 cows (at 1,000 each) and one year later receive 11,000 total even though the price of a cow now is 1,100. 
* **Increase in money supply:** Short term interest rate drops, but if increase in money supply is large, it may cause inflation, and that will raise interest rate in the long term
* **Inflation rate differential (X is a country):** $\text{Real FX}_{X, appreciation} = \text{Inflation}_X - \text{Nominal depreciation}_X$
* **Money supply:**  is comprised of 1. currency in circulation and 2. Demand deposits (checking account balances which are considered highly liquid)
  * central banks can issue new currency. In the US, the central bank is the Federal Reserve.
* **Money multiplier:** For each 100 physical dollars deposited into a bank account, a commercial bank can lend X amount of dollars to another party. Call $X/100$ the multiplier. So if X is 90, the multiplier is 90%.
* **Philips curve:** inflation rate / unemployment
* **Central banks try to balance: vigorous but sustainable GDP growth, low unemployment, low inflation, stable FX**


### Three tools of monetary policy 

- **Discount rate:** rate at which central bank can lend to commercial banks (apparently not useful for inflation targeting).
  - Decreate discount rate = increase money supply
  - Increase discount rate = decrease money supply. 
- **Reserve requirement on bank deposits:** increase in requirements decreases multiplier -> lowers money supply. 
- **Open market operations:** Central bank expands money supply by purchasing bonds or other assets from financial institutions. THIS affects the overnight bank rate and is the MAIN method of monetary policy.
  - [Read here from the Fed regarding OMO](https://www.federalreserve.gov/monetarypolicy/openmarket.htm)
* **overnight bank rate-aka fed funds rate:** Interest rate commercial banks charge one another for overnight lending. This borrowing of funds that are on deposit at the Fed though no lending or borrowing by the federal government is involved. Apparently very useful for affecting short term interest rates. This is what the Fed manages daily to affect ST interest rates and to a lesser degree the LT int rates.
* **Expectations** are paramount in determining the effectiveness of monetary policy
* **Liquidity trap:** when central bank tries to lower int rates but people keep absorbing the increase in money supply so rates do not drop
* **Fiscal policy spending** can be used to stimulate the economy, $GDP = C + I + G + EX - IM$. 
  * If government spending (G) increases, GDP increases and people may spend more (encouraging consumption). 
  * The Income multiplier works just like the money multiplier, such that after several iterations, the 100 increase in G results in a total increase in GDP of 500, assuming a multiplier of 80\% (100+80+64+....)
* **Fiscal Stimulus**
  * increase gov't spending -> increase demand -> increase in Nominal GDP via income multiplier
  * In preriods of high unemployment:
    * increase gov't spending -> increase demand -> Q increases so real GDP improves (recovery)
  * In preriods of full employment:
    * increase gov't spending -> increase demand -> P increases so you have inflation (overheating) 
  * In normal times:
    * increase gov't spending -> increase demand -> P and Q increase so you have inflation and increased GDP 
* **Leakages:** Affect the multiplier effects of fiscal stimulus. One example is the expectation people have that taxes will increase when a government runs a large deficit. This may in turn increase the savings rate of people and thus affect the multiplier. 

* **Keynesian** fiscal policy (public expenditure) is all about expectations
* Expectations strongly affect other macroeconomic variables
  * Example: If a bond trader expects interest rates to increase, she will sell the bonds. If many traders do the same, bond price goes down, and interest rates will go up
* The gold standard was enacted and essentially created a discipline since FX traders would ask for their money's worth of gold if they believed the value of the USD was below its gold equivalent
* After various iterations, by the 1990s the Fed adopted an implicit strategy to target inflation of 2% and was very adept at controlling the short term interest rate, i.e. the fed funds rate, by engaging in market operations.


### GDP accounting:
- **Value added:** at each step in production, value added = revenue - non labor costs. $\sum_i{\text{value added}_i} = \text{national GDP}$
- **Income:** value added must ultimately be allocated to members of the public. National GDP = sum (wages, salaries, int, dividends, rent, royalties, adjusted for dep & tax)
- **Expenditure (most widely used):** national GDP = sum of nation's market value of FINAL goods & services.
G = consumption + investment + gov expenditure + net exports (EX-IM) = C + S + T -Tr
  - ex: if a cafe in SF buys an italian cofee maker, a credit is added to I but a debit is added to IM so the effect is 0 since this product was produced abroad
Note: welfare payments, capital gain and purchase of used goods are not counted in either method
* **GNP:** counts output of a countries residents abroad. Think Toyota profits from Detroit plant goes to counting Japan's GNP
* GDP figures use a 'chained base year' method to account for changes in the product mix (new and products that no longer exist) and for change in usage patterns
I = S + (T - G - Tr) + (IM - EX), investment is funded from: private savings, gov savings, and borrowing from abroad
* **Balance of Payments (BOP):** accounting is another method. It provides a record of the country's cross-border transactions.
  * any cross border transaction has a debit(-) and a credit(+)
* **Interesting note:** when a country experiences a financial crisis, often times the BOP errors and omissions numbers increase in the months prior. The people in the know, start to move money out of their countries ahead of the crisis. 


### FX rates

* increased domestic demand for foreign products will likely depreciate its currency and and deteriorate current account balance
* increased domestic demand for a country's financial assets will likely cause the country's currency to appreciate even as its current account deteriorates. 
* increased foreign demand for a country's goods and services likely increase currency and current account balance
* Generally, sustained periods of current account deficit typically result in LT currency depreciation
* **Inflation and PPP:** via (law of one price). The country with higher inflation will see its currency depreciate. If US inflation is higher than UK, US will buy from UK (sell USD for GBP) for a better deal.
* Interest rates: traders view interest rates as most useful info to determine ST driver of FX rates. 
  * There is a completely different view here where the pop in FX rate (due to an increase in int rate) would have to be followed by enough of a depreciation for the 'law of one price' to be restored
* Financial Times article "The US Dollar: Defying experts" shows that in the 2005/2006 time frame many FX experts got the USD wrong. So using the facts of interest rate, inflation and demand for goods gives you a probabilistic view but is NOT definitive answer.
* FX rates are affected by many things: aggregate demand, currency interventions, int rate moves, deflation here, deflation there, financial panics, political crises, oil shocks, new technologies, changes in expectations and so forth
* **SUMMARY:** interest rates usually give ST view, Inflation usually gives MT view and current account imbalances give LT view


## Conclusion

Key concepts of macroeconomics: 
* **Output (account balance):** the amount that a country produces. GDP is the measure. If trading account deficit then country is either borrowing or spending excess savings. Trading and borrowing is tracked by the balance of payments (BOP)
* **Money (Price):**
  * in general, an increase in the money supply results in a decrease in interest rates, depreciation in FX rate and increase inflation (aggregate Price level)
  * when Price level rises, nominal values rise with it but real ones will not. Real Price levels are measured in terms of constant prices and thus control for inflation
  * governments manage money supply by changing: discount rate, reserve requirements, open market operations 
  * in US most of this money management is in terms of open market operations, i.e. buying/selling gov't bonds in the secondary market. This expands/contracts the money supply and affects the ST interest rate
  * Central bankers aim to:
    * maintain economic growth to a sustainable level
    * keep unemployment to a minimum
    * keep FX rates stable
    * keep interest rates at reasonable levels
    * keep inflation low
    * PRIMARY objective now seems to be to stabilize the price level by inflation targeting ~2\% which is achieved by managing interest rates (raise rates when inflation increases)
* **Expectations:** they can themselves become a driver of economic health. "If people become overly optimistic about their economic fortunes, they may drive demand above the productive capacity fo the economy"
* These macroeconomic rules provide a framework for asking questions and identifying which factors are at play (in times of departure from the norm or not) 


### Example
Below is a sample article regarding the August 22nd, 2018 FOMC meeting. Note how mention of several topics in this page appear on that article.

[FOMC Minutes Preview: The Yield Curve Is Back In Focus](../assets/FOMCMinutes20180822.pdf)
