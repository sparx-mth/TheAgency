"""
communication/__init__.py

Communication package initialization. Exports communication interfaces.
"""

from .comm_interface import CommunicationInterface
from .local_comm import LocalCommunication

__all__ = ['CommunicationInterface', 'LocalCommunication']