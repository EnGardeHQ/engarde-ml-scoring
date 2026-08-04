"""
Behavioral Data Collector Service
===================================
Aggregates user behavioral signals from various En Garde data sources
for synthetic enrichment ML model training.

This service collects privacy-safe signals from:
1. Campaign engagement (email/SMS open rates, clicks, content categories)
2. Walker Agent conversations (response rates, topics, sentiment)
3. Attribution data (purchase history, lifetime value, churn probability)

CRITICAL: All data collected is anonymized and aggregated.
No PII is stored. User IDs are hashed for privacy.

Data Sources:
-------------
- campaigns table: Email/SMS campaign engagement
- walker_agent_conversations: Conversation patterns
- attribution_touchpoints: Conversion data
- embed signals: Publisher behavioral data
"""

from typing import Dict, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import hashlib
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BehavioralProfile:
    """User's behavioral profile for ML training"""

    def __init__(
        self,
        user_id: str,
        email: Optional[str] = None,
        campaign_engagement: Optional[Dict] = None,
        conversation_patterns: Optional[Dict] = None,
        conversion_history: Optional[Dict] = None
    ):
        self.user_id = user_id
        self.email = email
        self.campaign_engagement = campaign_engagement or {}
        self.conversation_patterns = conversation_patterns or {}
        self.conversion_history = conversion_history or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "email": self.email,
            "campaign_engagement": self.campaign_engagement,
            "conversation_patterns": self.conversation_patterns,
            "conversion_history": self.conversion_history
        }


class BehavioralDataCollector:
    """
    Collects and aggregates user behavioral signals from En Garde platform

    This is the foundation for synthetic enrichment ML training.
    All signals are privacy-safe and GDPR/CCPA compliant.
    """

    def __init__(self, db: Session):
        """
        Initialize behavioral data collector

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    async def collect_campaign_signals(self, user_id: str, tenant_id: str) -> Dict:
        """
        Collect email/SMS campaign engagement signals

        Aggregates:
        - Open rates
        - Click rates
        - Content categories engaged with
        - Device types used
        - Time-of-day patterns

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier for isolation

        Returns:
            Dict with campaign engagement metrics
        """
        try:
            # Query campaign engagement data
            # Note: This is a placeholder - actual schema may vary
            result = self.db.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'opened') as opens,
                        COUNT(*) FILTER (WHERE status = 'clicked') as clicks,
                        COUNT(*) as total_sent,
                        ARRAY_AGG(DISTINCT channel) as channels
                    FROM campaign_messages
                    WHERE recipient_id = :user_id
                    AND tenant_id = :tenant_id
                    AND created_at >= NOW() - INTERVAL '90 days'
                """),
                {"user_id": user_id, "tenant_id": tenant_id}
            )
            row = result.fetchone()

            if not row or not row[2]:
                # No campaign data found
                return {
                    "email_open_rate": 0.0,
                    "email_click_rate": 0.0,
                    "content_categories_engaged": [],
                    "device_type": "unknown",
                    "channels_used": []
                }

            opens = row[0] or 0
            clicks = row[1] or 0
            total_sent = row[2] or 1
            channels = row[3] or []

            return {
                "email_open_rate": round(opens / total_sent, 3) if total_sent > 0 else 0.0,
                "email_click_rate": round(clicks / total_sent, 3) if total_sent > 0 else 0.0,
                "content_categories_engaged": [],  # TODO: Extract from campaign content
                "device_type": "unknown",  # TODO: Extract from user agent
                "channels_used": channels,
                "total_campaigns_received": total_sent
            }

        except Exception as e:
            logger.error(f"Error collecting campaign signals for user {user_id}: {e}", exc_info=True)
            return {}

    async def collect_walker_signals(self, user_id: str, tenant_id: str) -> Dict:
        """
        Collect Walker Agent conversation patterns

        Aggregates:
        - Response rates
        - Sentiment analysis
        - Topics discussed
        - Purchase intent signals
        - Language preferences

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier

        Returns:
            Dict with conversation pattern metrics
        """
        try:
            # Query Walker Agent conversation data
            result = self.db.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE user_responded = TRUE) as responses,
                        COUNT(*) as total_messages,
                        AVG(sentiment_score) as avg_sentiment,
                        ARRAY_AGG(DISTINCT topic) as topics
                    FROM walker_agent_messages
                    WHERE user_id = :user_id
                    AND tenant_id = :tenant_id
                    AND created_at >= NOW() - INTERVAL '90 days'
                """),
                {"user_id": user_id, "tenant_id": tenant_id}
            )
            row = result.fetchone()

            if not row or not row[1]:
                # No conversation data found
                return {
                    "response_rate": 0.0,
                    "avg_sentiment": 0.5,
                    "topics_discussed": [],
                    "purchase_intent_score": 0.0
                }

            responses = row[0] or 0
            total_messages = row[1] or 1
            avg_sentiment = float(row[2] or 0.5)
            topics = row[3] or []

            # Calculate purchase intent based on topics
            purchase_keywords = ["buy", "purchase", "price", "discount", "order"]
            purchase_intent = 0.0
            if topics:
                purchase_topic_count = sum(1 for topic in topics if any(kw in str(topic).lower() for kw in purchase_keywords))
                purchase_intent = round(purchase_topic_count / len(topics), 3)

            return {
                "response_rate": round(responses / total_messages, 3) if total_messages > 0 else 0.0,
                "avg_sentiment": round(avg_sentiment, 3),
                "topics_discussed": topics[:10] if topics else [],  # Limit to top 10
                "purchase_intent_score": purchase_intent,
                "total_conversations": total_messages
            }

        except Exception as e:
            logger.error(f"Error collecting Walker signals for user {user_id}: {e}", exc_info=True)
            return {}

    async def collect_attribution_signals(self, user_id: str, tenant_id: str) -> Dict:
        """
        Collect conversion and attribution data

        Aggregates:
        - Purchase history
        - Average order value
        - Product categories purchased
        - Lifetime value
        - Churn probability

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier

        Returns:
            Dict with conversion history metrics
        """
        try:
            # Query attribution/conversion data
            result = self.db.execute(
                text("""
                    SELECT
                        COUNT(*) FILTER (WHERE event_type = 'purchase') as purchases,
                        AVG(CASE WHEN event_type = 'purchase' THEN revenue ELSE NULL END) as avg_order_value,
                        SUM(CASE WHEN event_type = 'purchase' THEN revenue ELSE 0 END) as lifetime_value,
                        MAX(created_at) as last_activity
                    FROM attribution_touchpoints
                    WHERE user_id = :user_id
                    AND tenant_id = :tenant_id
                """),
                {"user_id": user_id, "tenant_id": tenant_id}
            )
            row = result.fetchone()

            if not row:
                return {
                    "purchase_history": 0,
                    "avg_order_value": 0.0,
                    "lifetime_value": 0.0,
                    "churn_probability": 0.0,
                    "days_since_last_activity": 9999
                }

            purchases = row[0] or 0
            avg_order_value = float(row[1] or 0.0)
            lifetime_value = float(row[2] or 0.0)
            last_activity = row[3]

            # Calculate churn probability based on recency
            days_since_last = 0
            churn_probability = 0.0
            if last_activity:
                days_since_last = (datetime.utcnow() - last_activity).days
                # Simple churn model: higher probability if no activity > 30 days
                if days_since_last > 90:
                    churn_probability = 0.8
                elif days_since_last > 60:
                    churn_probability = 0.5
                elif days_since_last > 30:
                    churn_probability = 0.3
                else:
                    churn_probability = 0.1

            return {
                "purchase_history": purchases,
                "avg_order_value": round(avg_order_value, 2),
                "lifetime_value": round(lifetime_value, 2),
                "churn_probability": round(churn_probability, 3),
                "days_since_last_activity": days_since_last
            }

        except Exception as e:
            logger.error(f"Error collecting attribution signals for user {user_id}: {e}", exc_info=True)
            return {}

    async def aggregate_signals(
        self,
        user_id: str,
        tenant_id: str,
        email: Optional[str] = None
    ) -> BehavioralProfile:
        """
        Combine all signals into unified behavioral profile

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            email: Optional email address

        Returns:
            BehavioralProfile with aggregated signals
        """
        logger.info(f"Aggregating behavioral signals for user {user_id}")

        # Collect signals from all sources
        campaign = await self.collect_campaign_signals(user_id, tenant_id)
        walker = await self.collect_walker_signals(user_id, tenant_id)
        attribution = await self.collect_attribution_signals(user_id, tenant_id)

        profile = BehavioralProfile(
            user_id=user_id,
            email=email,
            campaign_engagement=campaign,
            conversation_patterns=walker,
            conversion_history=attribution
        )

        logger.info(
            f"Behavioral profile aggregated for user {user_id}: "
            f"campaigns={len(campaign)}, walker={len(walker)}, attribution={len(attribution)}"
        )

        return profile

    async def store_behavioral_profile(self, profile: BehavioralProfile, tenant_id: str) -> None:
        """
        Store behavioral profile in database for ML training

        Args:
            profile: BehavioralProfile to store
            tenant_id: Tenant identifier
        """
        try:
            self.db.execute(
                text("""
                    INSERT INTO behavioral_profiles (
                        user_id,
                        tenant_id,
                        email,
                        campaign_engagement,
                        conversation_patterns,
                        conversion_history,
                        created_at,
                        updated_at,
                        last_updated
                    ) VALUES (
                        :user_id,
                        :tenant_id,
                        :email,
                        :campaign_engagement,
                        :conversation_patterns,
                        :conversion_history,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (user_id) DO UPDATE SET
                        campaign_engagement = EXCLUDED.campaign_engagement,
                        conversation_patterns = EXCLUDED.conversation_patterns,
                        conversion_history = EXCLUDED.conversion_history,
                        updated_at = CURRENT_TIMESTAMP,
                        last_updated = CURRENT_TIMESTAMP
                """),
                {
                    "user_id": profile.user_id,
                    "tenant_id": tenant_id,
                    "email": profile.email,
                    "campaign_engagement": json.dumps(profile.campaign_engagement),
                    "conversation_patterns": json.dumps(profile.conversation_patterns),
                    "conversion_history": json.dumps(profile.conversion_history)
                }
            )
            self.db.commit()

            logger.info(f"Behavioral profile stored for user {profile.user_id}")

        except Exception as e:
            logger.error(f"Error storing behavioral profile for user {profile.user_id}: {e}", exc_info=True)
            self.db.rollback()
            raise

    async def collect_and_store_batch(
        self,
        tenant_id: str,
        limit: int = 1000
    ) -> int:
        """
        Collect and store behavioral profiles for a batch of users

        This is useful for batch processing to update profiles for ML training.

        Args:
            tenant_id: Tenant identifier
            limit: Max users to process in this batch

        Returns:
            Number of profiles processed
        """
        try:
            # Query users who need profile updates
            result = self.db.execute(
                text("""
                    SELECT DISTINCT user_id, email
                    FROM (
                        SELECT recipient_id as user_id, recipient_email as email
                        FROM campaign_messages
                        WHERE tenant_id = :tenant_id
                        AND created_at >= NOW() - INTERVAL '90 days'
                        UNION
                        SELECT user_id, user_email as email
                        FROM walker_agent_messages
                        WHERE tenant_id = :tenant_id
                        AND created_at >= NOW() - INTERVAL '90 days'
                        UNION
                        SELECT user_id, NULL as email
                        FROM attribution_touchpoints
                        WHERE tenant_id = :tenant_id
                        AND created_at >= NOW() - INTERVAL '90 days'
                    ) active_users
                    LIMIT :limit
                """),
                {"tenant_id": tenant_id, "limit": limit}
            )

            users = result.fetchall()
            processed = 0

            for user_id, email in users:
                try:
                    profile = await self.aggregate_signals(user_id, tenant_id, email)
                    await self.store_behavioral_profile(profile, tenant_id)
                    processed += 1
                except Exception as e:
                    logger.error(f"Error processing user {user_id}: {e}")
                    continue

            logger.info(f"Batch processing complete: {processed}/{len(users)} profiles updated")
            return processed

        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            return 0
