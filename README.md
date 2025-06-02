This was my team's final algorithmic submission for the IMC Prosperity 3 competition.

Prosperity 3 was a worldwide algorithmic trading competition in which we traded assets against proprietary bots under a simulated environment.

We peaked at 340 algorithmic and 63 manual at the end of Round 3. This bot is not perfect, but I believe it is a strong and helpful framework for those who wish to compete next year.

The competition was composed of 5 rounds, where a new product was introduced each round (withholding the fifth round, in which the aforementioned proprietary trading bots were named and made accessible to leverage targeted strategies).

The products traded in each round were as follows:

Round 1: Kelp, Rainforest Resin, and Squid Ink

Round 2: Kelp, Rainforest Resin, and Squid Ink, Croissants, Jam, Djembes, Picnic Basket 1 (composed of 6 Croissants, 4 Jams, and 1 Djembe), and Picnic Basket 2 (composed of 4 Croissants and 2 Jams)

Round 3: Kelp, Rainforest Resin, and Squid Ink, Croissants, Jam, Djembes, Picnic Basket 1, Picnic Basket 2, Volcanic Rock, and Volcanic Rock Vouchers at varying strikes (analogue of a European Call option)

Round 4: Kelp, Rainforest Resin, and Squid Ink, Croissants, Jam, Djembes, Picnic Basket 1, Picnic Basket 2, Volcanic Rock, Volcanic Rock Vouchers, and Magnificent Macarons

Round 5: Same as Round 4 (no new products added, but simulated trading bots were named)

Each product had its own unique characteristics which changed how we approached trading them:

Rainforest Resin: Relatively stable, oscillating around a fair value of 10,000 sea shells throughout the entire competition.

Kelp: More volatile, but averaged a true value that made it possible to market-make with a spread.

Squid Ink: Very volatile with no true fair value. We traded this using mean reversion along a slope with z-score-based statistical arbitrage.

Volcanic Rock: Followed a similar random walk to Squid Ink; we employed the same strategy to trade it as well.

Volcanic Rock Vouchers: The value was determined by the underlying. As time approached expiry, the option was either valued at a premium or became worthless depending on whether it was ITM or OTM. We utilized a Black-Scholes inversion and solved for IV. We then used a polyfit on IV to trade the IV surface smile, indicating whether the market was optimistic or fearful for each voucher strike.

Magnificent Macarons: The sunlight index affected sugar prices and therefore was the greatest arbitrage indicator of current value versus fair value. We found the sunlight index to follow a day/night cycle and realized that when sunlight was greater or less than expected for that time of day, the price of Magnificent Macarons would fall or rise accordingly based on the availability or scarcity of sunlight. Importantly, the asset degrades in high sunlight and is expensive to store in refrigeration, so this must be taken into account. The asset may be traded with other bots or exported back to the island it was produced on, with import and export tariffs. We failed to properly account for this, so our conversion class is currently broken and leaked profit — but it can serve as an example.

Important to note: The Logger class was very helpful in selective logging with jsonpickle and was indispensable in the backtesting process.

I hope this was helpful. See you all next year!
