import logging
from typing import List, Dict, Any, Optional
import spacy
from spacy import displacy

from ..models.document import Entity, EntityType
from config.settings import settings


class EntityExtractor:
    """Entity extraction service for named entity recognition."""
    
    def __init__(self):
        """Initialize entity extractor."""
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the spaCy NLP model."""
        try:
            self.nlp = spacy.load(settings.SPACY_MODEL)
            self.logger.info(f"Loaded spaCy model: {settings.SPACY_MODEL}")
        except OSError:
            self.logger.error(f"spaCy model {settings.SPACY_MODEL} not found. Please install it with: python -m spacy download {settings.SPACY_MODEL}")
            raise
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            self._load_model()
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                # Map spaCy labels to our EntityType enum
                entity_type = self._map_spacy_label(ent.label_)
                
                if entity_type:  # Only include mapped entity types
                    entity = Entity(
                        text=ent.text,
                        label=ent.label_,
                        entity_type=entity_type,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=self._calculate_confidence(ent),
                        metadata={
                            "spacy_label": ent.label_,
                            "spacy_explanation": spacy.explain(ent.label_)
                        }
                    )
                    entities.append(entity)
            
            self.logger.debug(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text to process
            
        Returns:
            List of relationships between entities
        """
        if not self.nlp:
            self._load_model()
        
        try:
            doc = self.nlp(text)
            relationships = []
            
            # Simple relationship extraction based on dependency parsing
            for token in doc:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:  # Subject, direct object, prepositional object
                    head = token.head
                    
                    # Check if both token and head are part of named entities
                    token_ent = self._get_entity_for_token(token, doc.ents)
                    head_ent = self._get_entity_for_token(head, doc.ents)
                    
                    if token_ent and head_ent and token_ent != head_ent:
                        relationship = {
                            "subject": token_ent.text,
                            "subject_type": self._map_spacy_label(token_ent.label_),
                            "predicate": head.text,
                            "object": head_ent.text,
                            "object_type": self._map_spacy_label(head_ent.label_),
                            "relation_type": token.dep_,
                            "confidence": 0.7  # Basic confidence score
                        }
                        relationships.append(relationship)
            
            self.logger.debug(f"Extracted {len(relationships)} relationships from text")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}")
            return []
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """
        Map spaCy entity labels to our EntityType enum.
        
        Args:
            spacy_label: spaCy entity label
            
        Returns:
            Corresponding EntityType or None if not mapped
        """
        mapping = {
            # Person
            'PERSON': EntityType.PERSON,
            
            # Places
            'GPE': EntityType.PLACE,  # Geopolitical entity
            'LOC': EntityType.PLACE,  # Location
            'FAC': EntityType.BUILDING,  # Facility/Building
            
            # Organizations
            'ORG': EntityType.ORGANIZATION,
            
            # Events
            'EVENT': EntityType.EVENT,
            
            # Dates
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE,
        }
        
        return mapping.get(spacy_label)
    
    def _calculate_confidence(self, entity) -> float:
        """
        Calculate confidence score for an entity.
        
        Args:
            entity: spaCy entity object
            
        Returns:
            Confidence score between 0 and 1
        """
        # Basic confidence calculation based on entity properties
        confidence = 0.5  # Base confidence
        
        # Increase confidence for longer entities
        if len(entity.text) > 3:
            confidence += 0.1
        
        # Increase confidence for capitalized entities
        if entity.text.istitle():
            confidence += 0.1
        
        # Increase confidence for certain entity types
        high_confidence_types = ['PERSON', 'GPE', 'ORG']
        if entity.label_ in high_confidence_types:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _get_entity_for_token(self, token, entities):
        """
        Get the entity that contains a specific token.
        
        Args:
            token: spaCy token
            entities: List of spaCy entities
            
        Returns:
            Entity containing the token or None
        """
        for ent in entities:
            if ent.start <= token.i < ent.end:
                return ent
        return None
    
    def get_entity_summary(self, entities: List[Entity]) -> Dict[str, int]:
        """
        Get summary statistics for extracted entities.
        
        Args:
            entities: List of entities
            
        Returns:
            Dictionary with entity type counts
        """
        summary = {}
        for entity in entities:
            entity_type = entity.entity_type.value
            summary[entity_type] = summary.get(entity_type, 0) + 1
        
        return summary
    
    def visualize_entities(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Create HTML visualization of entities in text.
        
        Args:
            text: Input text
            output_path: Optional file path to save HTML
            
        Returns:
            HTML string with entity visualization
        """
        if not self.nlp:
            self._load_model()
        
        try:
            doc = self.nlp(text)
            
            # Generate HTML visualization
            html = displacy.render(doc, style="ent", jupyter=False)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                self.logger.info(f"Entity visualization saved to: {output_path}")
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error creating entity visualization: {str(e)}")
            return ""