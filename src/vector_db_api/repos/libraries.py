from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from vector_db_api.models.entities import Library

class LibraryRepo:
    def __init__(self) -> None:
        self.libraries: Dict[UUID, Library] = {}
    
    def add(self, lib: Library) -> None:
        now = datetime.utcnow()
        lib.created_at = now
        lib.updated_at = now
        self.libraries[lib.id] = lib
    
    def get(self, lib_id: UUID) -> Optional[Library]:
        library = self.libraries.get(lib_id)
        return library.model_copy(deep=True) if library else None
    
    def update_on_version(self, lib: Library, expected_version: int) -> bool:
        current_lib = self.libraries.get(lib.id)

        if current_lib is None or current_lib.version != expected_version:
            return False
        
        lib.version = expected_version + 1
        lib.updated_at = datetime.utcnow()
        self.libraries[lib.id] = lib
        return True
    
    def delete(self, lib_id: UUID) -> bool:
        return self.libraries.pop(lib_id, None) is not None
    
    def list(self) -> List[Library]:
        return [lib.model_copy(deep=True) for lib in self.libraries.values()]