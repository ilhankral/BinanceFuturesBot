"""
WebSocket Manager for Real-time Data from Binance
- Manages WebSocket connections to Binance Futures API
- Handles market data, account updates, and user data streams
- Provides real-time price, order and position updates
- Maintains data cache for fast access by trading bot
"""

import json
import asyncio
import websockets
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from urllib.parse import urlencode
import logging

# Configure logging
logger = logging.getLogger("WebSocketManager")


class BinanceWebSocketManager:
    """Manages WebSocket connections to Binance Futures API"""

    def __init__(self, symbols: List[str], api_key: str, api_secret: str, testnet: bool = False):
        """Initialize the WebSocket manager"""
        self.symbols = symbols
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Base URLs
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_base_url = "wss://stream.binancefuture.com/ws"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_base_url = "wss://fstream.binance.com/ws"

        # WebSocket connection objects
        self.market_ws = None
        self.user_ws = None
        self.listen_key = None

        # Flags
        self.running = False
        self.connected = False

        # Data caches
        self.price_cache = {}  # Symbol -> current price
        self.kline_cache = {}  # Symbol_timeframe -> latest klines
        self.orderbook_cache = {}  # Symbol -> current orderbook
        self.position_cache = {}  # Symbol -> current position
        self.account_cache = {}  # Account information
        self.order_cache = {}  # Client order ID -> order status

        # Subscription tracking
        self.active_subscriptions = set()

        # Task references
        self.tasks = []

    async def start(self):
        """Start the WebSocket manager"""
        if self.running:
            logger.warning("WebSocket manager already running")
            return

        self.running = True

        # Create tasks
        self.tasks = [
            asyncio.create_task(self._maintain_market_connection()),
            asyncio.create_task(self._maintain_user_connection()),
            asyncio.create_task(self._refresh_listen_key())
        ]

        logger.info("WebSocket manager started")

    async def stop(self):
        """Stop the WebSocket manager"""
        if not self.running:
            return

        self.running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Close connections
        if self.market_ws:
            await self.market_ws.close()
            self.market_ws = None

        if self.user_ws:
            await self.user_ws.close()
            self.user_ws = None

        # Reset state
        self.connected = False
        self.active_subscriptions.clear()

        logger.info("WebSocket manager stopped")

    async def _get_listen_key(self) -> Optional[str]:
        """Get a listen key for user data stream"""
        try:
            url = f"{self.base_url}/fapi/v1/listenKey"
            timestamp = int(time.time() * 1000)
            headers = {
                "X-MBX-APIKEY": self.api_key
            }

            # Create signature
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                f"timestamp={timestamp}".encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

            # Add signature to URL
            url = f"{url}?timestamp={timestamp}&signature={signature}"

            # Make request using websockets
            async with websockets.connect(f"wss://{url.split('://')[1]}") as ws:
                await ws.send("POST")
                response = await ws.recv()
                data = json.loads(response)
                return data.get("listenKey")

        except Exception as e:
            logger.error(f"Error getting listen key: {e}")
            return None

    async def _refresh_listen_key(self):
        """Refresh the listen key every 30 minutes"""
        while self.running:
            try:
                if self.listen_key:
                    url = f"{self.base_url}/fapi/v1/listenKey"
                    timestamp = int(time.time() * 1000)
                    headers = {
                        "X-MBX-APIKEY": self.api_key
                    }

                    # Create signature
                    signature = hmac.new(
                        self.api_secret.encode('utf-8'),
                        f"listenKey={self.listen_key}&timestamp={timestamp}".encode('utf-8'),
                        hashlib.sha256
                    ).hexdigest()

                    # Add signature to URL
                    url = f"{url}?listenKey={self.listen_key}&timestamp={timestamp}&signature={signature}"

                    # Make request using websockets
                    async with websockets.connect(f"wss://{url.split('://')[1]}") as ws:
                        await ws.send("PUT")
                        response = await ws.recv()
                        logger.debug("Listen key refreshed")

                # Wait 30 minutes
                await asyncio.sleep(30 * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error refreshing listen key: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute

    async def _maintain_market_connection(self):
        """Maintain connection to market data stream"""
        reconnect_delay = 1
        max_reconnect_delay = 300  # 5 minutes max

        while self.running:
            try:
                # Connect to WebSocket
                self.market_ws = await websockets.connect(self.ws_base_url)

                # Subscribe to streams for each symbol
                await self._subscribe_to_market_streams()

                # Set connected flag
                self.connected = True
                logger.info("Connected to market data stream")

                # Process messages
                async for message in self.market_ws:
                    if not self.running:
                        break

                    await self._process_market_message(message)

                # If we get here, the connection was closed
                self.connected = False
                logger.warning("Market data stream connection closed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                logger.error(f"Market WebSocket error: {e}")
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")

                # Wait before reconnecting
                await asyncio.sleep(reconnect_delay)

                # Exponential backoff for reconnect delay
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _maintain_user_connection(self):
        """Maintain connection to user data stream"""
        reconnect_delay = 1
        max_reconnect_delay = 300  # 5 minutes max

        while self.running:
            try:
                # Get listen key if we don't have one
                if not self.listen_key:
                    self.listen_key = await self._get_listen_key()
                    if not self.listen_key:
                        logger.error("Failed to get listen key")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue

                # Connect to WebSocket with listen key
                url = f"{self.ws_base_url}/{self.listen_key}"
                self.user_ws = await websockets.connect(url)

                logger.info("Connected to user data stream")

                # Process messages
                async for message in self.user_ws:
                    if not self.running:
                        break

                    await self._process_user_message(message)

                # If we get here, the connection was closed
                logger.warning("User data stream connection closed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"User WebSocket error: {e}")
                logger.info(f"Reconnecting in {reconnect_delay} seconds...")

                # Reset listen key
                self.listen_key = None

                # Wait before reconnecting
                await asyncio.sleep(reconnect_delay)

                # Exponential backoff for reconnect delay
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _subscribe_to_market_streams(self):
        """Subscribe to market data streams"""
        if not self.market_ws:
            logger.error("Cannot subscribe, market WebSocket not connected")
            return

        # Create subscription message for all symbols and streams
        streams = []

        # Add streams for each symbol
        for symbol in self.symbols:
            symbol_lower = symbol.lower()

            # Ticker stream (price updates)
            streams.append(f"{symbol_lower}@ticker")

            # Kline streams (candlestick data)
            for interval in ["1m", "5m", "15m", "1h", "4h", "1d"]:
                streams.append(f"{symbol_lower}@kline_{interval}")

            # Orderbook stream (top 10 bids/asks)
            streams.append(f"{symbol_lower}@depth10@100ms")

        # Create subscription message
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }

        # Send subscription message
        await self.market_ws.send(json.dumps(subscribe_msg))

        # Track active subscriptions
        self.active_subscriptions.update(streams)

        logger.info(f"Subscribed to {len(streams)} market data streams")

    async def _process_market_message(self, message: str):
        """Process a message from the market data stream"""
        try:
            data = json.loads(message)

            # Skip subscription response messages
            if "result" in data:
                return

            # Get event time
            event_time = datetime.fromtimestamp(data.get("E", 0) / 1000)

            # Process based on stream type
            if "stream" in data:
                stream = data["stream"]
                stream_data = data["data"]

                # Process ticker updates
                if "@ticker" in stream:
                    symbol = stream_data.get("s")
                    price = float(stream_data.get("c", 0))  # Close price

                    # Update price cache
                    if symbol:
                        self.price_cache[symbol] = {
                            "price": price,
                            "bid": float(stream_data.get("b", 0)),
                            "ask": float(stream_data.get("a", 0)),
                            "high": float(stream_data.get("h", 0)),
                            "low": float(stream_data.get("l", 0)),
                            "volume": float(stream_data.get("v", 0)),
                            "timestamp": event_time
                        }

                # Process kline updates
                elif "@kline" in stream:
                    parts = stream.split("@")
                    symbol = parts[0].upper()
                    interval = parts[1].replace("kline_", "")

                    kline = stream_data.get("k", {})

                    # Only process completed klines
                    if kline.get("x", False):  # x = is closed
                        key = f"{symbol}_{interval}"

                        # Create kline data
                        kline_data = {
                            "time": datetime.fromtimestamp(kline.get("t", 0) / 1000),
                            "open": float(kline.get("o", 0)),
                            "high": float(kline.get("h", 0)),
                            "low": float(kline.get("l", 0)),
                            "close": float(kline.get("c", 0)),
                            "volume": float(kline.get("v", 0)),
                            "trades": int(kline.get("n", 0)),
                            "interval": interval
                        }

                        # Update kline cache
                        if key not in self.kline_cache:
                            self.kline_cache[key] = []

                        # Add kline and keep only the most recent 500
                        self.kline_cache[key].append(kline_data)
                        self.kline_cache[key] = self.kline_cache[key][-500:]

                # Process orderbook updates
                elif "@depth" in stream:
                    parts = stream.split("@")
                    symbol = parts[0].upper()

                    # Update orderbook cache
                    self.orderbook_cache[symbol] = {
                        "bids": [[float(price), float(qty)] for price, qty in stream_data.get("bids", [])],
                        "asks": [[float(price), float(qty)] for price, qty in stream_data.get("asks", [])],
                        "timestamp": event_time
                    }

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in market message: {message}")
        except Exception as e:
            logger.error(f"Error processing market message: {e}")

    async def _process_user_message(self, message: str):
        """Process a message from the user data stream"""
        try:
            data = json.loads(message)
            event_type = data.get("e")

            # Account update event
            if event_type == "ACCOUNT_UPDATE":
                update = data.get("a", {})

                # Process balance updates
                balances = update.get("B", [])
                for balance in balances:
                    asset = balance.get("a")
                    wallet_balance = float(balance.get("wb", 0))

                    # Update account cache
                    if "balances" not in self.account_cache:
                        self.account_cache["balances"] = {}

                    self.account_cache["balances"][asset] = wallet_balance

                # Process position updates
                positions = update.get("P", [])
                for position in positions:
                    symbol = position.get("s")
                    amount = float(position.get("pa", 0))
                    entry_price = float(position.get("ep", 0))

                    # Update position cache
                    if amount != 0:
                        self.position_cache[symbol] = {
                            "symbol": symbol,
                            "amount": amount,
                            "entry_price": entry_price,
                            "mark_price": float(position.get("mp", 0)),
                            "unrealized_pnl": float(position.get("up", 0)),
                            "side": "LONG" if amount > 0 else "SHORT",
                            "update_time": datetime.now()
                        }
                    elif symbol in self.position_cache:
                        # Position closed
                        del self.position_cache[symbol]

            # Order update event
            elif event_type == "ORDER_TRADE_UPDATE":
                order = data.get("o", {})
                symbol = order.get("s")
                client_order_id = order.get("c")
                order_status = order.get("X")

                # Update order cache
                self.order_cache[client_order_id] = {
                    "symbol": symbol,
                    "order_id": order.get("i"),
                    "client_order_id": client_order_id,
                    "side": order.get("S"),
                    "type": order.get("o"),
                    "price": float(order.get("p", 0)),
                    "quantity": float(order.get("q", 0)),
                    "status": order_status,
                    "time": datetime.fromtimestamp(order.get("T", 0) / 1000),
                    "filled_qty": float(order.get("l", 0)),
                    "avg_price": float(order.get("ap", 0))
                }

                # Log order status changes
                logger.info(f"Order {client_order_id} {order_status}: {symbol} {order.get('S')} {order.get('o')}")

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in user message: {message}")
        except Exception as e:
            logger.error(f"Error processing user message: {e}")

    def is_connected(self) -> bool:
        """Check if the WebSocket connections are active"""
        return self.connected

    async def reconnect(self):
        """Force a reconnection of all WebSockets"""
        await self.stop()
        await asyncio.sleep(1)
        await self.start()

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a symbol from the cache"""
        if symbol in self.price_cache:
            return self.price_cache[symbol]["price"]
        return None

    def get_recent_klines(self, symbol: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get recent klines from the cache"""
        key = f"{symbol}_{interval}"
        if key in self.kline_cache:
            return self.kline_cache[key][-limit:]
        return []

    def get_orderbook(self, symbol: str) -> Dict:
        """Get the current orderbook for a symbol"""
        if symbol in self.orderbook_cache:
            return self.orderbook_cache[symbol]
        return {"bids": [], "asks": [], "timestamp": None}

    def get_balance(self, asset: str = "USDT") -> float:
        """Get the balance for an asset"""
        if "balances" in self.account_cache and asset in self.account_cache["balances"]:
            return self.account_cache["balances"][asset]
        return 0.0

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get the current position for a symbol"""
        if symbol in self.position_cache:
            return self.position_cache[symbol]
        return None

    def get_open_positions(self) -> Dict[str, Dict]:
        """Get all open positions"""
        return self.position_cache

    def get_order_status(self, client_order_id: str) -> Optional[Dict]:
        """Get the status of an order by client order ID"""
        if client_order_id in self.order_cache:
            return self.order_cache[client_order_id]
        return None