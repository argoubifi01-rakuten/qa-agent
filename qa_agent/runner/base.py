from abc import ABC, abstractmethod
from qa_agent.models import TestCase, RunResult


class Runner(ABC):
    @abstractmethod
    def run(self, test_case: TestCase) -> RunResult:
        """Execute one test case and return its result."""
