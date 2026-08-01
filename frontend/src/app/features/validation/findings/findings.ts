import { Component, inject, OnInit, signal, computed } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { FormBuilder, ReactiveFormsModule } from '@angular/forms';
import { MatTableModule } from '@angular/material/table';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatSelectModule } from '@angular/material/select';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatPaginatorModule, PageEvent } from '@angular/material/paginator';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatDialogModule, MatDialog } from '@angular/material/dialog';
import { MatChipsModule } from '@angular/material/chips';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatExpansionModule } from '@angular/material/expansion';
import { NgClass } from '@angular/common';
import { ValidationService } from '../../../core/services/validation.service';
import { AuthService } from '../../../core/services/auth.service';
import { NotificationService } from '../../../core/services/notification.service';
import { Finding, FindingFilters, Severity, FindingStatus } from '../../../core/models';
import { ResolveDialogComponent } from '../resolve-dialog/resolve-dialog';

@Component({
  selector: 'app-findings',
  imports: [
    ReactiveFormsModule, NgClass,
    MatTableModule, MatFormFieldModule, MatInputModule, MatSelectModule,
    MatButtonModule, MatIconModule, MatPaginatorModule, MatProgressSpinnerModule,
    MatDialogModule, MatChipsModule, MatTooltipModule, MatExpansionModule,
  ],
  templateUrl: './findings.html',
  styleUrl: './findings.css',
})
export class FindingsComponent implements OnInit {
  private readonly svc    = inject(ValidationService);
  private readonly route  = inject(ActivatedRoute);
  private readonly fb     = inject(FormBuilder);
  private readonly dialog = inject(MatDialog);
  private readonly notify = inject(NotificationService);
  readonly auth           = inject(AuthService);

  readonly loading  = signal(true);
  readonly findings = signal<Finding[]>([]);

  // Pagination
  readonly pageSize  = signal(50);
  readonly pageIndex = signal(0);
  readonly offset    = computed(() => this.pageIndex() * this.pageSize());

  readonly columns = ['severity', 'rule_id', 'domain', 'field', 'usubjid', 'row_index', 'status', 'actions'];

  readonly severities: Severity[]     = ['CRIT', 'HIGH', 'MED', 'LOW'];
  readonly statuses: FindingStatus[]  = ['open', 'resolved', 'waived'];
  readonly domains = ['DM', 'VS', 'AE', 'CM', 'EX', 'LB', 'DS', 'EG', 'QS', 'RS', 'PR', 'CROSS', 'GENERAL'];

  readonly filters = this.fb.nonNullable.group({
    domain:   [''],
    severity: ['' as Severity | ''],
    status:   ['' as FindingStatus | ''],
    usubjid:  [''],
  });

  private _jobId?:   string;
  private _studyId?: string;

  ngOnInit(): void {
    this._jobId   = this.route.snapshot.queryParamMap.get('job_id') ?? undefined;
    this._studyId = this.route.snapshot.queryParamMap.get('study_id') ?? undefined;
    this.load();
  }

  load(): void {
    this.loading.set(true);
    const f = this.filters.getRawValue();
    const params: FindingFilters = {
      job_id:   this._jobId,
      study_id: this._studyId,
      domain:   f.domain || undefined,
      severity: (f.severity as Severity) || undefined,
      status:   (f.status as FindingStatus) || undefined,
      usubjid:  f.usubjid || undefined,
      offset:   this.offset(),
      limit:    this.pageSize(),
    };

    this.svc.listFindings(params).subscribe({
      next: findings => { this.findings.set(findings); this.loading.set(false); },
      error: () => this.loading.set(false),
    });
  }

  applyFilters(): void {
    this.pageIndex.set(0);
    this.load();
  }

  clearFilters(): void {
    this.filters.reset();
    this.pageIndex.set(0);
    this.load();
  }

  onPage(e: PageEvent): void {
    this.pageIndex.set(e.pageIndex);
    this.pageSize.set(e.pageSize);
    this.load();
  }

  openResolve(finding: Finding): void {
    this.dialog.open(ResolveDialogComponent, { width: '480px', data: { finding } })
      .afterClosed().subscribe(resolved => { if (resolved) this.load(); });
  }

  sevClass(sev: string): string { return `chip-${sev}`; }
  statusClass(s: string): string { return `chip-${s}`; }
  sevIcon(sev: string): string {
    return { CRIT: 'dangerous', HIGH: 'warning', MED: 'info', LOW: 'check_circle' }[sev] ?? 'help';
  }
}
