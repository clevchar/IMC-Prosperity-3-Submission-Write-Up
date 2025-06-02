# 2025-IMC-Prosperity-Submission
This was my teams final algorithmic submission for the IMC Prosperity 3 competition. 

Prosperity 3 was a worldwide algorithmic trading competition in which we traded assets against proprietary bots under a simulated environment.

We peaked at 340 algrothmic and 63 manual at the end of round 3. This bot is not perfect but I believe is a strong and helpful framework for those who wish to compete next year.

The competion was composed of 5 rounds where a new product was introduced each round (witholding the fifth round in which the aforementioneed proprietary trading bots were named and made accesible to leverage targeted strategies.

The products traded in each round were as follows:

-Round 1: Kelp, Rainforest Resin, and Squid Ink

-Round 2: Kelp, Rainforest Resin, and Squid Ink, Croissonts, Jam, Djembes, Picnic Basket 1 (Composed of 6 Croissonts, 4 Jams, and 1 Djembe), and Picnic Basket 2 (Composed of 4 Croissonts and 2 Jams)

-Round 3: Kelp, Rainforest Resin, and Squid Ink, Croissonts, Jam, Djembes, Picnic Basket 1 (Composed of 6 Croissonts, 4 Jams, and 1 Djembe), and Picnic Basket 2 (Composed of 4 Croissonts and 2 Jams), Volcanic Rock, and Volcanic Rock Vouchers at varying strikes (Analogue of a European Call option)

-Round 4: Kelp, Rainforest Resin, and Squid Ink, Croissonts, Jam, Djembes, Picnic Basket 1 (Composed of 6 Croissonts, 4 Jams, and 1 Djembe), and Picnic Basket 2 (Composed of 4 Croissonts and 2 Jams), Volcanic Rock, and Volcanic Rock Vouchers (Analogue of a European Call option), Magnificent Macroons

-Round 5: Kelp, Rainforest Resin, and Squid Ink, Croissonts, Jam, Djembes, Picnic Basket 1 (Composed of 6 Croissonts, 4 Jams, and 1 Djembe), and Picnic Basket 2 (Composed of 4 Croissonts and 2 Jams), Volcanic Rock, and Volcanic Rock Vouchers (Analogue of a European Call option), Magnificent Macroons (No new products added but simulated trading bot's named)

Each product had it's own unique charachteristics which changed how we approached trading them.

-Rainforest Resin: Relatively stable oscillating around a fair value of 10,000 sea shells throughout the entire competition.

-Kelp: More volatile but averaged a true that made it possible to market make with a spread.

-Squid Ink: Very volatile with no true fair value, we traded this using mean reversion along a slope with z-score based statistical arbitrage.

-Volcanic Rock: Followed a similar random walk to squid ink, we employed the same strategy to trade it aswell. 

-Volcanic Rock Vouchers: Since the value was determined by the underlying, that is as time is closer to expiry the option is valued at premium or worthless depending on if it's ITM or OTM. We utilized a Black Scholes inversion and solved for IV. We then used a polyfit on IV to trade the IV surface smile, indicating wether the market was optimistic or frightful of each voucher strike.

-Magnificent Macrons: The sunlight index affected sugar price and therefore were the greatest arbitrage indicators in current value versus fair value. We found sunlight index to be a day and night cycle and realized when sunlight was greater or less than the expected at that time of day the price of Magnificent Macrons would fall or rise according based on the availibilty or scarcity of sunlight. Importantly, the asset degrades as well in high sunlight and is expensive to store in refrigeration so this must be taken into account. The asset may be traded with other bots or back to the island it was produced on with import and export tariffs. We failed to properly account for this so currently our conversion class is broken and leaked profit, but can serve as an example.

Important to note: The logger class was very helpful in selective logging with jsonpcikle and was indespensible in the backtesting process. 

I hope this was helpful. See you all next year!

