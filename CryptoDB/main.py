
from cryptofeed import FeedHandler
from cryptofeed.backends.kafka import BookKafka, TradeKafka
from cryptofeed.defines import L2_BOOK, TRADES, L3_BOOK, L1_BOOK
from cryptofeed.exchanges import Coinbase, Binance, Bitfinex, Bitmex


def main():
    f = FeedHandler()
    cbs = {
           TRADES: TradeKafka(),
           L1_BOOK: BookKafka(),
           L2_BOOK: BookKafka(),
           L3_BOOK: BookKafka()
          }

    Bitfinex.symbols()
    # Add bitcoin data to Feed
    f.add_feed(Bitfinex(max_depth=25, channels=[TRADES, L2_BOOK], symbols=['BTC-USD'], callbacks=cbs))

    # If you wish to collect lv2 order data and trade data from Binance
    f.add_feed(Binance(max_depth=25, channels=[TRADES, L2_BOOK], symbols=['BTC-BUSD'], callbacks=cbs))
    f.add_feed(Bitmex(max_depth=25, channels=[TRADES, L2_BOOK], symbols=['BTC-USD-PERP'], callbacks=cbs))

    f.run()


if __name__ == '__main__':
    main()